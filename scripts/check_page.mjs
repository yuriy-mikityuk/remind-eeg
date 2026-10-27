#!/usr/bin/env node
// Open the local app in headless Chrome, go through it, and fail on any page error.
//
// The app tests check the strings the server sends. This runs the page itself, so an
// error that stops app.js (a const used before its declaration once did) fails CI.
//
// Usage: node scripts/check_page.mjs http://127.0.0.1:8799/
// The app should run with the mock source. Needs Node 22 (or Node 20 with
// --experimental-websocket) and Chrome; CHROME overrides the browser binary.

import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const appUrl = new URL(process.argv[2] || "http://127.0.0.1:8765/");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

if (typeof WebSocket === "undefined") {
  console.error("check_page needs a WebSocket client: run it on Node 22, or Node 20 with --experimental-websocket");
  process.exit(2);
}

function chromeBinary() {
  const candidates = [
    process.env.CHROME,
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (spawnSync(candidate, ["--version"], { stdio: "ignore" }).status === 0) {
      return candidate;
    }
  }
  console.error("no Chrome found; set CHROME to the browser binary");
  process.exit(2);
}

const port = 9300 + Math.floor(Math.random() * 600);
const profile = mkdtempSync(join(tmpdir(), "check-page-"));
const chrome = spawn(
  chromeBinary(),
  [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${port}`,
    ...(process.platform === "linux" ? ["--no-sandbox"] : []),
    "about:blank"
  ],
  { stdio: "ignore" }
);

const problems = [];
// Set while the page is cut off on purpose; its failed requests are expected then.
let offline = false;
// While set, ui-state answers carry this build, as if the app had updated itself.
let fakeBuild = null;
let socket = null;
let nextId = 1;
const pending = new Map();

async function finish(code) {
  try {
    socket?.close();
  } catch (error) {
    // already closed
  }
  // Closed through its own protocol, Chrome and its helpers are done with the profile
  // once it exits; a plain kill leaves helpers writing to it for a while.
  const exited = once(chrome, "exit");
  try {
    const { webSocketDebuggerUrl } = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    const browser = new WebSocket(webSocketDebuggerUrl);
    await new Promise((resolve) => browser.addEventListener("open", resolve, { once: true }));
    browser.send(JSON.stringify({ id: 1, method: "Browser.close" }));
  } catch (error) {
    chrome.kill();
  }
  await Promise.race([exited, sleep(5000)]);
  if (chrome.exitCode === null) {
    chrome.kill();
  }
  try {
    // Its helper processes can still be writing for a moment after it exits.
    rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
  } catch (error) {
    console.error(`could not remove ${profile}: ${error.message}`);
  }
  process.exit(code);
}

async function connect() {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" });
      return await response.json();
    } catch (error) {
      await sleep(500);
    }
  }
  throw new Error("Chrome did not open its debugging port");
}

function send(method, params = {}) {
  return new Promise((resolve) => {
    const id = nextId;
    nextId += 1;
    pending.set(id, resolve);
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const reply = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (reply.result?.exceptionDetails) {
    throw new Error(reply.result.exceptionDetails.exception?.description || reply.result.exceptionDetails.text);
  }
  return reply.result?.result?.value;
}

// Each check is asked again until it holds or 10 s pass, so a slow runner is not a failure.
async function step(name, action, check) {
  await action();
  let verdict = null;
  for (let attempt = 0; attempt < 40 && verdict !== true; attempt += 1) {
    await sleep(250);
    try {
      verdict = await evaluate(check);
    } catch (error) {
      verdict = String(error.message || error); // the page is still loading
    }
  }
  if (verdict !== true) {
    problems.push(`${name}: ${verdict}`);
  }
  console.log(`${verdict === true ? "ok  " : "FAIL"} ${name}`);
}

async function withFakeBuild({ requestId, responseStatusCode, responseHeaders }) {
  try {
    const { result } = await send("Fetch.getResponseBody", { requestId });
    const payload = JSON.parse(result.base64Encoded ? Buffer.from(result.body, "base64").toString() : result.body);
    payload.build = fakeBuild;
    await send("Fetch.fulfillRequest", {
      requestId,
      responseCode: responseStatusCode,
      responseHeaders,
      body: Buffer.from(JSON.stringify(payload)).toString("base64")
    });
  } catch (error) {
    await send("Fetch.continueRequest", { requestId });
  }
}

const click = (selector) => () => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const visit = (path) => () => send("Page.navigate", { url: new URL(path, appUrl).href });

try {
  const target = await connect();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    } else if (message.method === "Fetch.requestPaused") {
      withFakeBuild(message.params);
    } else if (message.method === "Runtime.exceptionThrown") {
      const details = message.params.exceptionDetails;
      problems.push(`uncaught: ${details.exception?.description || details.text}`);
    } else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
      problems.push(`console.error: ${message.params.args.map((arg) => arg.value ?? arg.description).join(" ")}`);
    } else if (message.method === "Log.entryAdded" && message.params.entry.level === "error") {
      if (offline && message.params.entry.source === "network") {
        return;
      }
      problems.push(`${message.params.entry.source}: ${message.params.entry.text} ${message.params.entry.url || ""}`.trim());
    }
  });
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Log.enable");

  await step(
    "the page starts and shows the version",
    visit("/"),
    `document.querySelector("#app-version").textContent.startsWith("Version") || "the page script did not get as far as the first ui-state"`
  );
  await step(
    "connecting the mock headband shows its four channels",
    click("#connect-button"),
    `document.querySelectorAll("#contact-list .contact-row").length === 4 || "contact rows: " + document.querySelectorAll("#contact-list .contact-row").length`
  );
  await step(
    "the Morning tab opens",
    click("#tab-morning"),
    `(!document.querySelector("#screen-morning").hidden && document.querySelector("#screen-record").hidden) || "Morning tab not shown"`
  );
  await step(
    "picking another retest wave with nothing typed asks nothing",
    () =>
      evaluate(`(() => {
        window.confirmCalls = 0;
        window.confirm = () => {
          window.confirmCalls += 1;
          return false;
        };
        const wave = document.querySelector("#retest-wave");
        wave.value = "day7";
        // A picked option fires input, then change, and both bubble to the form.
        wave.dispatchEvent(new Event("input", { bubbles: true }));
        wave.dispatchEvent(new Event("change", { bubbles: true }));
      })()`),
    `(window.confirmCalls === 0 && document.querySelector("#retest-wave").value === "day7") ||
      "asked " + window.confirmCalls + " time(s), wave " + document.querySelector("#retest-wave").value`
  );
  await step(
    "the Recordings tab lists recordings",
    click("#tab-recordings"),
    `(!document.querySelector("#screen-recordings").hidden && document.querySelector("#history-list").children.length > 0) || "Recordings list empty"`
  );
  await step(
    "back on the Record tab, the modes are there",
    click("#tab-record"),
    `(!document.querySelector("#screen-record").hidden && document.querySelectorAll("#modes .mode").length === 5) || "modes: " + document.querySelectorAll("#modes .mode").length`
  );
  await step(
    "a reload keeps the Recordings tab",
    async () => {
      await click("#tab-recordings")();
      await sleep(300);
      await send("Page.reload");
    },
    `!document.querySelector("#screen-recordings").hidden || "reload lost the tab"`
  );
  const network = (cutOff) =>
    send("Network.emulateNetworkConditions", { offline: cutOff, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await step(
    "cut off from the app, the page says it is out of date",
    async () => {
      await click("#tab-record")();
      await send("Network.enable");
      offline = true;
      await network(true);
    },
    `!document.querySelector("#app-lost").hidden || "no banner after the app went silent"`
  );
  await step(
    "and drops that once the app answers again",
    () => network(false),
    `document.querySelector("#app-lost").hidden || "the banner stayed"`
  );
  offline = false;
  await step(
    "and does not flash it after the page was asleep",
    async () => {
      await evaluate(`(() => {
        window.bannerShown = false;
        new MutationObserver(() => {
          window.bannerShown = window.bannerShown || !document.querySelector("#app-lost").hidden;
        }).observe(document.querySelector("#app-lost"), { attributes: true, attributeFilter: ["hidden"] });
      })()`);
      // Frozen, its timers stop the way they do while the Mac sleeps.
      await send("Page.setWebLifecycleState", { state: "frozen" });
      await sleep(7000);
      await send("Page.setWebLifecycleState", { state: "active" });
      await sleep(2000);
    },
    `window.bannerShown === false || "the banner flashed after waking"`
  );
  await step(
    "a new build of the app reloads the page",
    async () => {
      await visit("/")();
      await sleep(1500);
      await evaluate("window.notReloaded = true");
      fakeBuild = "next";
      await send("Fetch.enable", { patterns: [{ urlPattern: "*/api/muse/ui-state*", requestStage: "Response" }] });
    },
    // Done only once the reloaded page has taken "next" as its build, so the next step
    // cannot end the fake while the new page's first poll is still held.
    `(window.notReloaded !== true && document.querySelector("#app-version").textContent.includes("build next")) ||
      "the page stayed on the old build"`
  );
  // The tab draws the night's forms once they have loaded, over anything typed into
  // them before, and counts them as saved then. Only a checkout with a night in
  // data/protocol shows this; CI has none, so there the forms are never drawn.
  await step(
    "the Morning tab has loaded its nights",
    click("#tab-morning"),
    `!document.querySelector("#morning-empty").hidden || document.querySelector("#dream-status").textContent !== "" ||
      "the Morning tab did not finish loading"`
  );
  await step(
    "but not over a dream report being written",
    async () => {
      await evaluate(`(() => {
        window.notReloaded = true;
        const text = document.querySelector("#dream-text");
        text.value = "a staircase";
        text.dispatchEvent(new Event("input", { bubbles: true }));
      })()`);
      // The real build is not "next": to the page the app has updated again.
      await send("Fetch.disable");
      fakeBuild = null;
      await sleep(3000);
    },
    `(window.notReloaded === true && document.querySelector("#dream-text").value === "a staircase" &&
      document.querySelector("#app-version").textContent.includes("once the Morning answers are saved")) ||
      "the update reloaded over the dream report, or did not say it waits"`
  );
  await step(
    "the meditation guide loads",
    visit("/guide.html"),
    `document.querySelectorAll("[data-lang]").length > 0 || "guide did not render"`
  );
} catch (error) {
  problems.push(String(error?.stack || error));
}

if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s) on the page:`);
  problems.forEach((problem) => console.error(`- ${problem}`));
  await finish(1);
}
console.log("\nno page errors");
await finish(0);
