const appTitle = document.querySelector("#app-title");
const sourceLabel = document.querySelector("#source-label");
const deviceName = document.querySelector("#device-name");
const deviceConnectionAge = document.querySelector("#device-connection-age");
const deviceAddress = document.querySelector("#device-address");
const deviceSource = document.querySelector("#device-source");
const deviceLastPacket = document.querySelector("#device-last-packet");
const deviceBattery = document.querySelector("#device-battery");
const scanResult = document.querySelector("#scan-result");
const errorBox = document.querySelector("#error-box");
const appVersion = document.querySelector("#app-version");
const appFrame = document.querySelector(".app");
const appLostBanner = document.querySelector("#app-lost");
const scanButton = document.querySelector("#scan-button");
const connectButton = document.querySelector("#connect-button");
const startSessionButton = document.querySelector("#start-session-button");
const startNightButton = document.querySelector("#start-night-button");
const startMeditationButton = document.querySelector("#start-meditation-button");
const meditationForm = document.querySelector("#meditation-form");
const meditationCancel = document.querySelector("#med-cancel");
const meditationChime = document.querySelector("#med-chime");
const meditationPanel = document.querySelector("#meditation-panel");
const meditationNow = document.querySelector("#meditation-now");
const meditationSound = document.querySelector("#meditation-sound");
const meditationBlocks = document.querySelector("#meditation-blocks");
const analyzeMeditationButton = document.querySelector("#analyze-meditation-button");
const openMeditationReport = document.querySelector("#open-meditation-report");
const meditationAnalysisText = document.querySelector("#meditation-analysis-text");
const startSeriesButton = document.querySelector("#start-series-button");
const seriesForm = document.querySelector("#series-form");
const startCalibrationButton = document.querySelector("#start-calibration-button");
const calibrationForm = document.querySelector("#calibration-form");
const calibrationSteps = document.querySelector("#calibration-steps");
const calibrationPanel = document.querySelector("#calibration-panel");
const calibrationNow = document.querySelector("#calibration-now");
const calibrationReportButton = document.querySelector("#calibration-report-button");
const openCalibrationReport = document.querySelector("#open-calibration-report");
const calibrationReportText = document.querySelector("#calibration-report-text");
const stopRecordingButton = document.querySelector("#stop-recording-button");
const cancelStartButton = document.querySelector("#cancel-start-button");
const disconnectButton = document.querySelector("#disconnect-button");
const recordHint = document.querySelector("#record-hint");
const recordReport = document.querySelector("#record-report");
const recordReportText = document.querySelector("#record-report-text");
const recordReportPath = document.querySelector("#record-report-path");
const buildReportButton = document.querySelector("#build-report-button");
const openReportLink = document.querySelector("#open-report-link");
const backToMenuButton = document.querySelector("#back-to-menu-button");
const recordingStrip = document.querySelector("#recording-strip");
const recordingStatus = document.querySelector("#recording-status");
const recordingKind = document.querySelector("#recording-kind");
const recordingElapsed = document.querySelector("#recording-elapsed");
const recordingBattery = document.querySelector("#recording-battery");
const recordingLastEvent = document.querySelector("#recording-last-event");
const recordingPolar = document.querySelector("#recording-polar");
const recordingProgressBar = document.querySelector("#recording-progress-bar");
const recordingStarted = document.querySelector("#recording-started");
const recordingEnds = document.querySelector("#recording-ends");
const recordingEeg = document.querySelector("#recording-eeg");
const recordingLost = document.querySelector("#recording-lost");
const recordingReconnects = document.querySelector("#recording-reconnects");
const modes = document.querySelector("#modes");
const workNote = document.querySelector("#work-note");
const tabRecord = document.querySelector("#tab-record");
const tabRecordings = document.querySelector("#tab-recordings");
const screenRecord = document.querySelector("#screen-record");
const screenRecordings = document.querySelector("#screen-recordings");
// The card each kind of start highlights while it waits for contact.
const modeCards = {
  night: document.querySelector("#mode-night"),
  session: document.querySelector("#mode-session"),
  meditation: document.querySelector("#mode-meditation"),
  series: document.querySelector("#mode-series"),
  calibration: document.querySelector("#mode-calibration")
};
const polarOption = document.querySelector("#polar-option");
const withPolarCheckbox = document.querySelector("#with-polar-checkbox");
const contactSummary = document.querySelector("#contact-summary");
const contactList = document.querySelector("#contact-list");
const allGoodCheck = document.querySelector("#all-good-check");
const sessionStrip = document.querySelector("#session-strip");
const sessionStatus = document.querySelector("#session-status");
const sessionElapsed = document.querySelector("#session-elapsed");
const sessionWarnings = document.querySelector("#session-warnings");
const sessionStreamRate = document.querySelector("#session-stream-rate");
const warningLogSection = document.querySelector("#warning-log-section");
const activeWarningStatus = document.querySelector("#active-warning-status");
const warningLog = document.querySelector("#warning-log");
const diagNotificationsRate = document.querySelector("#diag-notifications-rate");
const diagEegRowsRate = document.querySelector("#diag-eeg-rows-rate");
const diagEegEffectiveRate = document.querySelector("#diag-eeg-effective-rate");
const diagDecodeErrors = document.querySelector("#diag-decode-errors");
const diagUnknownTags = document.querySelector("#diag-unknown-tags");
const diagLastPacketAge = document.querySelector("#diag-last-packet-age");

const contactChannels = ["TP9", "AF7", "AF8", "TP10"];
const contactHistory = Object.fromEntries(contactChannels.map((channel) => [channel, []]));

let latestState = {};
let latestContact = {};
let latestGate = {};
let latestDiagnostics = {};
let latestRecording = {};
// A start waiting for the contact gate: { label, card, url, body, done }.
let pendingStart = null;
let pendingSending = false;
let startError = "";
// The output folder of a finished recording put away with "Back to menu"; kept over a reload.
let dismissedRecording = readDismissedRecording();
let loadedBuild = null;

const stateText = {
  disconnected: "Disconnected",
  scanning: "Scanning",
  connecting: "Connecting",
  connected: "Connected",
  error: "Error"
};

const reasonText = {
  no_recent_samples: "stale samples",
  source_disconnected: "disconnected",
  stale_snapshot: "stale samples",
  stale_contact: "stale contact",
  low_coverage: "low coverage",
  mild_noise: "mild noise",
  hard_artifact: "artifact",
  clipping: "clipping",
  flatline: "flatline",
  non_finite: "bad samples"
};

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderState(state) {
  latestState = { ...latestState, ...state };
  if (latestState.connection_state !== "connected" && !pendingSending) {
    pendingStart = null;
  }
  renderSourceBadge(latestState.source || "unknown");

  errorBox.hidden = !latestState.error_message;
  errorBox.textContent = latestState.error_message || "";
  renderActions();

  renderAppTitle();
  renderDeviceCard();
  renderSessionStrip();
  renderWarningLog();
}

// The page asks for the state every second. With no answer for a few seconds the
// app has quit, crashed or is restarting, and what the page shows is out of date.
// Only time spent waiting on a poll counts: after the Mac slept, or a background tab
// had its timers held back, the last answer is old but the app may be fine.
const APP_SILENT_MS = 5000;
let waitingSince = null;

async function refreshUiState() {
  if (waitingSince == null) {
    waitingSince = Date.now();
  }
  let payload;
  try {
    payload = await requestJson("/api/muse/ui-state");
  } catch (error) {
    return; // the banner says so once the app stays silent
  }
  waitingSince = null;
  renderAppLost();
  renderUiState(payload);
}

function renderAppLost() {
  const lost = waitingSince != null && Date.now() - waitingSince > APP_SILENT_MS;
  appLostBanner.hidden = !lost;
  appFrame.classList.toggle("lost", lost);
}

function renderUiState(payload) {
  // The server restarts itself on a new main (app --auto-update); reload so
  // the page runs the matching JS, but not over Morning answers still being written.
  let updateWaits = false;
  if (payload.build) {
    if (loadedBuild == null) {
      loadedBuild = payload.build;
    } else if (payload.build !== loadedBuild) {
      if (!morningUnsaved()) {
        window.location.reload();
        return;
      }
      updateWaits = true;
    }
  }
  appVersion.textContent = [
    payload.version && `Version ${payload.version}`,
    payload.build && `build ${payload.build}`,
    updateWaits && "the update loads once the Morning answers are saved"
  ]
    .filter(Boolean)
    .join(" · ");
  latestDiagnostics = { source_diagnostics: payload.source_diagnostics || null };
  latestRecording = payload.recording || {};
  latestContact = payload.contact || {};
  renderState(payload.state || {});
  renderContact(payload.contact || {});
  renderGate(payload.gate || {});
  renderDiagnostics(latestDiagnostics);
  renderRecording(latestRecording);
  reloadSeriesAfterRecording(latestRecording);
  renderAppTitle();
}

// While a recording runs the app has handed BLE to the recorder, so the
// headband state comes from the recorder's contact snapshot.
function recorderStreaming() {
  return (
    Boolean(latestRecording.active) &&
    latestContact.connection_state === "connected" &&
    !latestContact.stale
  );
}

function displayConnection() {
  if (latestRecording.active) {
    return recorderStreaming() ? "connected" : "connecting";
  }
  return latestState.connection_state || "disconnected";
}

function connectionLabel() {
  if (latestRecording.active) {
    return recorderStreaming() ? "Streaming to recorder" : "Waiting for recorder data";
  }
  const connection = latestState.connection_state || "disconnected";
  return stateText[connection] || connection;
}

function recordingTitle(recording) {
  if (recording.calibration) {
    return "Calibration run";
  }
  if (recording.meditation) {
    return "Meditation";
  }
  return recording.kind === "night" ? "Night recording" : "One-hour session";
}

// A finished recording stays on the Record tab until "Back to menu" puts it away.
function finishedRecordingShown() {
  const state = latestRecording.state;
  const outputDir = latestRecording.output_dir;
  return (state === "completed" || state === "failed") && Boolean(outputDir) && outputDir !== dismissedRecording;
}

function renderAppTitle() {
  // A tab in the background still says that a recording runs and for how long.
  document.title =
    latestRecording && latestRecording.active
      ? `● ${formatDuration(latestRecording.elapsed_seconds)} · ${recordingTitle(latestRecording)} · REMind`
      : "REMind";
  if (latestRecording && latestRecording.active) {
    appTitle.textContent = recordingTitle(latestRecording);
  } else if (finishedRecordingShown()) {
    appTitle.textContent = latestRecording.state === "completed" ? "Recording finished" : "Recording ended";
  } else {
    appTitle.textContent = "Start a recording";
  }
}

function renderActions() {
  const connection = latestState.connection_state || "disconnected";
  const recordingActive = Boolean(latestRecording.active);
  const isAmused = latestState.source === "amused";

  // While a detached recording runs, the app has released BLE; collapse setup controls.
  scanButton.hidden = connection === "connected" || recordingActive;
  connectButton.hidden = connection === "connected" || recordingActive;
  disconnectButton.hidden = connection !== "connected" || recordingActive;

  // The modes stay in view, so it is clear what each one records; they start nothing until connected.
  const canRecord = connection === "connected" && !recordingActive;
  const finished = finishedRecordingShown();
  modes.hidden = recordingActive || finished;
  workNote.hidden = modes.hidden;
  workNote.textContent =
    connection === "connected"
      ? `Each one starts once all four channels have been good for ${stabilitySeconds()} s.`
      : "Connect the headband to start one.";
  if (!canRecord) {
    meditationForm.hidden = true;
    seriesForm.hidden = true;
    calibrationForm.hidden = true;
  }
  polarOption.hidden = !(canRecord && isAmused) || modes.hidden;
  stopRecordingButton.hidden = !(
    recordingActive && latestRecording.state !== "stopping" && latestRecording.state !== "finishing"
  );

  // Recording needs the live headband. A start waits for good contact itself.
  const recordEnabled = canRecord && isAmused && !pendingStart;
  cancelStartButton.hidden = !(canRecord && pendingStart);
  Object.entries(modeCards).forEach(([key, card]) => {
    card.classList.toggle("waiting", Boolean(pendingStart) && pendingStart.card === key);
  });
  for (const form of [meditationForm, seriesForm, calibrationForm]) {
    form.querySelector('button[type="submit"]').disabled = Boolean(pendingStart);
  }
  startSessionButton.disabled = !recordEnabled;
  startNightButton.disabled = !recordEnabled;
  startMeditationButton.disabled = !recordEnabled;
  startSeriesButton.disabled = !recordEnabled;
  startCalibrationButton.disabled = !recordEnabled;

  connectButton.classList.toggle("primary", connection !== "connected");
  connectButton.disabled = connection === "connecting";
  scanButton.disabled = connection === "scanning" || connection === "connecting";
  disconnectButton.disabled = connection === "disconnected";
}

function renderScanResult() {
  const scan = latestState.scan;
  const connection = latestState.connection_state || "disconnected";
  const show = (Boolean(scan) || connection === "scanning") && connection !== "connected" && !latestRecording.active;
  scanResult.hidden = !show;
  if (!show) {
    return;
  }
  if (connection === "scanning" || !scan) {
    scanResult.textContent = "Looking for the headband...";
    return;
  }
  if (scan.failed) {
    scanResult.textContent = `Scan failed (${scan.error || "Bluetooth error"}). Is Bluetooth on for this Mac?`;
    return;
  }
  const devices = latestState.devices || [];
  if (devices.length === 0) {
    scanResult.textContent =
      "No Muse found. Turn the headband on (hold the button until the lights run), close the Muse phone app so it releases Bluetooth, then try again.";
    return;
  }
  const best = devices[0];
  const signal = numberOrNull(best.rssi);
  const signalText = signal == null || signal <= -100 ? "" : `, signal ${Math.round(signal)} dBm`;
  if (scan.configured_found === false) {
    scanResult.textContent = `Found ${devices.length} Muse device(s), but not the configured headband (${shortAddress(scan.configured_address)}). Is another Muse nearby?`;
    return;
  }
  scanResult.textContent = `${best.name || "Muse"} is on and in range${signalText}. Press Connect Muse.`;
}

function shortAddress(address) {
  const text = String(address || "");
  return text.length > 12 ? `${text.slice(0, 8)}...` : text;
}

function renderSourceBadge(source) {
  const live = source === "amused";
  sourceLabel.textContent = live ? "LIVE Muse" : "MOCK";
  sourceLabel.className = `source-badge ${live ? "live" : "mock"}`;
}

function renderDeviceCard() {
  const connection = latestState.connection_state || "disconnected";
  const device = latestState.device || {};
  const diagnostics = latestDiagnostics.source_diagnostics || {};
  const candidates = latestState.devices || [];

  if (device.name || device.address) {
    deviceName.textContent = device.name || "Muse";
  } else if (latestRecording.active) {
    // The app handed the headband to the recorder and no longer knows its name.
    deviceName.textContent = "Headband";
  } else if (candidates.length > 0) {
    deviceName.textContent = candidates[0].name || "Muse";
  } else {
    deviceName.textContent = "No headset selected";
  }

  deviceConnectionAge.textContent =
    connection === "connected" && !latestRecording.active
      ? `Connected ${formatDuration(latestState.connected_elapsed_seconds)}`
      : connectionLabel();
  deviceConnectionAge.className = `connection-pill ${connectionPillState(connection)}`;
  deviceAddress.textContent = device.address || "-";
  deviceSource.textContent = latestState.source || "unknown";
  deviceLastPacket.textContent =
    latestState.source === "mock" ? "n/a" : formatAge(diagnostics.last_packet_age_seconds);
  const battery = numberOrNull(
    latestRecording.active ? latestRecording.battery_percent : latestState.battery_percent
  );
  deviceBattery.textContent = battery == null ? "-" : `${Math.round(battery)}%`;
  deviceBattery.classList.toggle("low", battery != null && battery < 30);
  renderScanResult();
}

function connectionPillState(connection) {
  if (latestRecording.active) {
    return "recording";
  }
  if (connection === "connected") {
    return "connected";
  }
  if (connection === "scanning" || connection === "connecting") {
    return "busy";
  }
  return connection === "error" ? "error" : "";
}

function renderContact(snapshot) {
  latestContact = snapshot;
  const channels = snapshot.channels || {};
  const badChannels = [];

  contactList.innerHTML = "";
  contactChannels.forEach((channel) => {
    const state = channels[channel] || {
      channel,
      status: "missing",
      fill: 0,
      reason_codes: ["no_recent_samples"]
    };
    const fill = clamp(Number(state.fill) || 0, 0, 1);
    const status = state.status || "missing";
    const reason = primaryReason(state.reason_codes || []);
    updateContactHistory(channel, fill);

    const segment = document.querySelector(`.segment-fill[data-channel="${channel}"]`);
    if (segment) {
      segment.style.strokeDasharray = `${fill} 1`;
      segment.classList.remove("missing", "poor", "fair", "good");
      segment.classList.add(status);
      segment.querySelector("title").textContent = `${channel}: ${status}, ${formatPercent(fill)}`;
    }
    if (status !== "good") {
      badChannels.push(`${channel} ${status}${reason ? ` (${reason})` : ""}`);
    }

    contactList.appendChild(contactRow(channel, status, fill, reason, state.reason_codes || []));
  });

  // Without a connection the rows would only say "missing" four times.
  contactList.hidden = snapshot.connection_state !== "connected";
  // An SVG element has no hidden property; setting one leaves the attribute in place.
  allGoodCheck.toggleAttribute("hidden", !snapshot.all_good);
  if (snapshot.stale) {
    contactSummary.textContent = "Contact data stale";
  } else if (snapshot.connection_state !== "connected") {
    contactSummary.textContent = "Connect the headband to see its contact";
  } else if (snapshot.all_good) {
    contactSummary.textContent = "All four channels good";
  } else {
    contactSummary.textContent = badChannels.length > 0 ? `Adjust ${badChannels.join(", ")}` : "Checking contact";
  }
  contactSummary.classList.toggle(
    "adjust",
    !snapshot.all_good && !snapshot.stale && snapshot.connection_state === "connected" && badChannels.length > 0
  );
}

function contactRow(channel, status, fill, reason, reasonCodes) {
  const row = document.createElement("li");
  row.className = `contact-row ${status}`;

  const name = document.createElement("span");
  name.className = "channel-name";
  name.textContent = channel;

  const statusCell = document.createElement("span");
  statusCell.className = "channel-status";
  const statusText = document.createElement("span");
  statusText.textContent = titleCase(status);
  statusCell.appendChild(statusText);
  if (reason) {
    const reasonTextNode = document.createElement("small");
    reasonTextNode.textContent = reason;
    statusCell.appendChild(reasonTextNode);
  }

  const fillCell = document.createElement("span");
  fillCell.className = "channel-fill";
  fillCell.textContent = formatPercent(fill);

  const sparkline = contactSparkline(channel);

  row.title = `${channel}: ${status}. ${reasonCodes.join(", ")}`;
  row.append(name, statusCell, fillCell, sparkline);
  return row;
}

function contactSparkline(channel) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "contact-sparkline");
  svg.setAttribute("viewBox", "0 0 56 22");
  svg.setAttribute("aria-hidden", "true");

  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", sparklinePoints(contactHistory[channel] || []));
  svg.appendChild(line);
  return svg;
}

function renderGate(gate) {
  latestGate = gate;
  sendPendingStartWhenReady();
  renderActions();
  renderAppTitle();
  renderSessionStrip();
  renderWarningLog();
}

function renderRecording(recording) {
  latestRecording = recording || {};
  const active = Boolean(latestRecording.active);
  const state = latestRecording.state;
  const connection = latestState.connection_state || "disconnected";
  const isAmused = latestState.source === "amused";

  recordingStrip.hidden = !active;
  if (active) {
    recordingStatus.textContent =
      state === "launching"
        ? "Starting"
        : state === "stopping"
        ? "Stopping"
        : state === "finishing"
        ? "Finishing (stopping the H10)"
        : "Recording";
    recordingKind.textContent = latestRecording.preset || "";
    recordingElapsed.textContent = formatDuration(latestRecording.elapsed_seconds);
    const fraction = numberOrNull(latestRecording.progress_fraction);
    recordingProgressBar.style.width = `${(clamp(fraction || 0, 0, 1) * 100).toFixed(1)}%`;
    const started = numberOrNull(latestRecording.started_at_seconds);
    const duration = numberOrNull(latestRecording.duration_seconds);
    recordingStarted.textContent = started == null ? "" : `Started ${formatTimeOfDay(started)}`;
    recordingEnds.textContent = started == null || duration == null ? "" : `Ends about ${formatTimeOfDay(started + duration)}`;
    const battery = numberOrNull(latestRecording.battery_percent);
    recordingBattery.textContent = battery == null ? "--" : `${Math.round(battery)}%`;
    // While recording, the diagnostics are the recorder's own heartbeat.
    const decoder = (latestDiagnostics.source_diagnostics || {}).decoder || {};
    recordingEeg.textContent = formatRate(decoder.eeg_effective_sample_rate_hz);
    const lost = subpacketTotal(decoder.lost_subpackets);
    recordingLost.textContent = lost == null ? "--" : String(lost);
    recordingReconnects.textContent = String(numberOrNull(latestRecording.reconnect_attempts) || 0);
    recordingLastEvent.textContent = latestRecording.last_event
      ? String(latestRecording.last_event).replaceAll("_", " ")
      : "streaming";
    const polarText = polarStatusText(latestRecording.polar);
    recordingPolar.hidden = !polarText;
    recordingPolar.textContent = polarText ? ` · ${polarText}` : "";
  }

  stopRecordingButton.textContent = state === "stopping" ? "Stopping" : "Stop recording";
  stopRecordingButton.disabled = state === "stopping";

  // Once put away, the page is back to the modes; the recording stays on the Recordings tab.
  const finished = finishedRecordingShown();
  recordReport.hidden = !finished;
  if (finished) {
    const report = latestRecording.report || { state: "none" };
    const ended = state === "completed" ? "Recording finished." : "Recording ended without a summary.";
    // A meditation or a calibration run has its own analysis in the panel above. The REM
    // report is for sleep: after those it stays on the Recordings tab unless one is under way.
    const ownAnalysis = Boolean((latestRecording.meditation && latestRecording.meditation.plan) || latestRecording.calibration);
    const offerRem = !ownAnalysis || report.state !== "none";
    const reportText = {
      none: "Build the REM report for it?",
      running: "Building the report...",
      ready: "Report ready.",
      failed: `Report failed, see ${report.log_path || "report.log"}. You can also run:`
    };
    recordReportText.textContent = offerRem ? `${ended} ${reportText[report.state] || ""}` : ended;
    buildReportButton.hidden = !offerRem || report.state === "ready";
    buildReportButton.disabled = report.state === "running";
    buildReportButton.textContent = report.state === "failed" ? "Try again" : "Build report";
    openReportLink.hidden = !(report.state === "ready" && report.url);
    if (report.url) {
      openReportLink.href = report.url;
    }
    recordReportPath.hidden = report.state !== "failed";
    recordReportPath.textContent = latestRecording.report_command || "";
  }

  // The hint speaks only when there is something to say: a start waiting, a failed start, a mock source.
  let hint = "";
  if (connection === "connected" && !active && !finished) {
    if (!isAmused) {
      hint = "Recording needs the live Muse source (run the app with --source amused).";
    } else if (pendingStart) {
      hint = pendingStartText();
    } else if (startError) {
      hint = `Could not start: ${startError}`;
    }
  }
  recordHint.hidden = !hint;
  recordHint.textContent = hint;

  const shown = active || finished ? latestRecording : {};
  renderMeditation(shown);
  renderCalibration(shown);
  renderActions();
  renderAppTitle();
}

function renderDiagnostics(diagnostics) {
  latestDiagnostics = diagnostics;
  const source = diagnostics.source_diagnostics || {};
  const decoder = source.decoder || {};
  const rollingEeg = numberOrNull(decoder.eeg_rolling_sample_rate_hz);
  const effectiveEeg = numberOrNull(decoder.eeg_effective_sample_rate_hz);

  if (latestState.source === "mock") {
    diagNotificationsRate.textContent = "n/a";
    diagEegRowsRate.textContent = "mock";
    diagEegEffectiveRate.textContent = "mock";
    diagDecodeErrors.textContent = "0";
    diagUnknownTags.textContent = "0";
    diagLastPacketAge.textContent = "n/a";
  } else {
    diagNotificationsRate.textContent = formatRate(
      decoder.rolling_notifications_per_second ?? decoder.notifications_per_second,
      "/s"
    );
    diagEegRowsRate.textContent = formatRate(rollingEeg ?? effectiveEeg, "rows/s");
    diagEegEffectiveRate.textContent = formatRate(effectiveEeg);
    diagDecodeErrors.textContent = String(decoder.decode_errors ?? 0);
    diagUnknownTags.textContent = formatUnknownTags(decoder.unknown_tag_counts || {});
    diagLastPacketAge.textContent = formatAge(source.last_packet_age_seconds);
  }

  renderDeviceCard();
  renderSessionStrip();
}

function renderSessionStrip() {
  const session = latestState.session || {};
  const gateState = latestGate.state;
  const running = session.running || gateState === "starting" || gateState === "running";
  sessionStrip.hidden = !running;
  if (!running) {
    return;
  }

  const decoder = (latestDiagnostics.source_diagnostics || {}).decoder || {};
  const streamRate = decoder.eeg_rolling_sample_rate_hz ?? decoder.eeg_effective_sample_rate_hz;
  sessionStatus.textContent = gateState === "starting" ? "Starting session" : "Session running";
  sessionElapsed.textContent = formatDuration(session.elapsed_seconds);
  sessionWarnings.textContent = `contact warnings: ${session.contact_warning_count || 0}`;
  sessionStreamRate.textContent =
    latestState.source === "mock" && streamRate == null
      ? "stream: mock"
      : `stream: ${formatRate(streamRate)}`;
}

function renderWarningLog() {
  const session = latestState.session || {};
  const events = session.contact_warning_events || [];
  const active = session.active_contact_warning;
  const running = session.running || latestGate.state === "starting" || latestGate.state === "running";

  warningLogSection.hidden = !running && events.length === 0 && !active;
  if (warningLogSection.hidden) {
    activeWarningStatus.hidden = true;
    activeWarningStatus.textContent = "";
    warningLog.innerHTML = "";
    return;
  }

  activeWarningStatus.hidden = !active;
  activeWarningStatus.textContent = active
    ? `${active.channels.join(", ")} for ${formatSeconds(active.elapsed_seconds)}`
    : "";

  const completedEvents = events.filter(
    (event) => event.kind !== "contact_drop" || event.duration_seconds != null
  );
  warningLog.innerHTML = "";
  if (completedEvents.length === 0 && !active) {
    const item = document.createElement("li");
    item.textContent = "No contact warnings";
    warningLog.appendChild(item);
    return;
  }

  [...completedEvents].reverse().forEach((event) => {
    const item = document.createElement("li");
    item.textContent = warningEventText(event);
    warningLog.appendChild(item);
  });
}

scanButton.addEventListener("click", async () => {
  renderState({ ...latestState, connection_state: "scanning" });
  await requestJson("/api/muse/scan", { method: "POST" });
  await refreshUiState();
});

connectButton.addEventListener("click", async () => {
  renderState({ ...latestState, connection_state: "connecting" });
  await requestJson("/api/muse/connect", { method: "POST" });
  await refreshUiState();
});

function polarStatusText(polar) {
  if (!polar) {
    return "";
  }
  const heart = polar.heart_rate_bpm != null && polar.contact !== false ? ` · HR ${polar.heart_rate_bpm}` : "";
  const age = numberOrNull(polar.data_age_seconds);
  switch (polar.state) {
    case "connected":
      return polar.contact === false ? "H10: no skin contact" : `H10: streaming${heart}`;
    case "stale":
      return `H10: no data for ${age == null ? "?" : Math.round(age)} s`;
    case "reconnecting":
      return "H10: reconnecting";
    case "failed":
      return "H10: failed (see polar/record-polar.log)";
    case "stopped":
      return "H10: stopped";
    default:
      return "H10: connecting";
  }
}

// Remember the Polar choice between sessions (per browser, best effort).
try {
  withPolarCheckbox.checked = window.localStorage.getItem("withPolar") === "1";
} catch (error) {
  withPolarCheckbox.checked = false;
}
withPolarCheckbox.addEventListener("change", () => {
  try {
    window.localStorage.setItem("withPolar", withPolarCheckbox.checked ? "1" : "0");
  } catch (error) {
    // storage unavailable; the checkbox still works for this page
  }
});

// --- start once the contact is good -------------------------------------------
// A start button arms the backend contact gate, and the start request goes out
// once the gate has seen all four channels good for its stability window. The
// backend refuses a start before that, so the wait is not only in the page.

function stabilitySeconds() {
  const seconds = numberOrNull(latestGate.required_stability_seconds);
  return seconds == null ? 5 : Math.round(seconds * 10) / 10;
}

function pendingStartText() {
  const label = pendingStart.label;
  if (latestGate.state === "ready_countdown") {
    const left = Math.max(1, Math.ceil(stabilitySeconds() - (numberOrNull(latestGate.stable_for_seconds) || 0)));
    return `Contact good. The ${label} starts in ${left} s if it stays that way.`;
  }
  if (latestGate.state === "ready") {
    return `Starting the ${label}...`;
  }
  return `The ${label} starts once all four channels have been good for ${stabilitySeconds()} s.`;
}

async function startWhenReady(start) {
  startError = "";
  pendingStart = start;
  renderRecording(latestRecording);
  try {
    renderGate(await requestJson("/api/session/record/arm", { method: "POST" }));
  } catch (error) {
    pendingStart = null;
    startError = error.message;
  }
  renderRecording(latestRecording);
}

function sendPendingStartWhenReady() {
  if (!pendingStart || pendingSending || latestGate.state !== "ready") {
    return;
  }
  const start = pendingStart;
  pendingSending = true;
  fetch(start.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(start.body)
  })
    .then(async (response) => {
      const payload = await response.json().catch(() => ({}));
      if (response.status === 409 && payload.gate) {
        return; // the contact slipped in between; the gate counts again
      }
      if (pendingStart === start) {
        pendingStart = null;
      }
      await start.done(response, payload);
    })
    .catch((error) => {
      if (pendingStart === start) {
        pendingStart = null;
      }
      startError = error.message;
    })
    .finally(() => {
      pendingSending = false;
      refreshUiState();
    });
}

function startRecording(kind) {
  startWhenReady({
    label: kind === "night" ? "night recording" : "session",
    card: kind === "night" ? "night" : "session",
    url: "/api/session/record",
    body: { kind, with_polar: withPolarCheckbox.checked },
    done: (response, payload) => {
      if (!response.ok) {
        startError = payload.error || `${response.status} ${response.statusText}`;
      }
    }
  });
}

async function cancelPendingStart() {
  if (!pendingStart) {
    return;
  }
  pendingStart = null;
  try {
    renderGate(await requestJson("/api/session/record/disarm", { method: "POST" }));
  } finally {
    await refreshUiState();
  }
}

cancelStartButton.addEventListener("click", cancelPendingStart);

startSessionButton.addEventListener("click", () => startRecording("session"));
startNightButton.addEventListener("click", () => startRecording("night"));

buildReportButton.addEventListener("click", async () => {
  buildReportButton.disabled = true;
  try {
    await requestJson("/api/session/report", { method: "POST" });
  } finally {
    await refreshUiState();
  }
});

function readDismissedRecording() {
  try {
    return window.localStorage.getItem("dismissedRecording");
  } catch (error) {
    return null;
  }
}

backToMenuButton.addEventListener("click", () => {
  dismissedRecording = latestRecording.output_dir;
  try {
    window.localStorage.setItem("dismissedRecording", dismissedRecording);
  } catch (error) {
    // it just shows again after a reload
  }
  renderRecording(latestRecording);
});

stopRecordingButton.addEventListener("click", async () => {
  stopRecordingButton.disabled = true;
  stopRecordingButton.textContent = "Stopping";
  try {
    await requestJson("/api/session/record/stop", { method: "POST" });
  } finally {
    await refreshUiState();
  }
});

disconnectButton.addEventListener("click", async () => {
  pendingStart = null;
  await requestJson("/api/muse/disconnect", { method: "POST" });
  await refreshUiState();
});

// --- guided meditation -------------------------------------------------------

const meditationState = { outputDir: null, items: new Map(), lastPhase: null, anchor: null };
let audioContext = null;
// Set by a click that tried to unlock blocked sound: the next click mutes even if the browser
// kept it blocked, so the Sound button cannot get stuck on "Enable sound".
let unlockTried = false;

function readChimeSetting() {
  try {
    return window.localStorage.getItem("meditationChime") !== "0";
  } catch (error) {
    return true;
  }
}

let chimeSetting = readChimeSetting();

function chimeEnabled() {
  return chimeSetting;
}

function setChimeEnabled(enabled) {
  chimeSetting = enabled;
  try {
    window.localStorage.setItem("meditationChime", enabled ? "1" : "0");
  } catch (error) {
    // the setting just won't survive a reload
  }
}

function ensureAudio() {
  if (!audioContext) {
    const Context = window.AudioContext || window.webkitAudioContext;
    audioContext = Context ? new Context() : null;
    if (audioContext) {
      audioContext.addEventListener("statechange", () => {
        if (audioRunning()) {
          unlockTried = false;
        }
        renderSoundButton();
      });
    }
  }
  if (audioContext && audioContext.state !== "running") {
    audioContext.resume().catch(() => {});
  }
}

// Suspended (no click yet after a reload, or Safari interrupted it) counts as blocked.
function audioRunning() {
  return Boolean(audioContext) && audioContext.state === "running";
}

// Browsers only allow sound after a click; any click on the page unlocks it again
// after a reload in the middle of a session.
document.addEventListener("click", () => {
  if (latestRecording.meditation && latestRecording.active) {
    ensureAudio();
  }
});

function playChime(count) {
  if (!chimeEnabled() || !audioContext) {
    return;
  }
  const start = audioContext.currentTime + 0.05;
  for (let index = 0; index < count; index += 1) {
    const time = start + index * 0.9;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 528;
    gain.gain.setValueAtTime(0.0001, time);
    gain.gain.exponentialRampToValueAtTime(0.08, time + 0.08);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + 1.6);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(time);
    oscillator.stop(time + 1.7);
  }
}

// The voice language, for the block changes here and for the calibration run on the Mac. Until
// one is picked in a form, the browser's own language decides.
function readVoiceLanguage() {
  try {
    const stored = window.localStorage.getItem("voiceLanguage");
    if (stored === "en" || stored === "ru") {
      return stored;
    }
  } catch (error) {
    // no storage: the browser language decides
  }
  const browser = (navigator.languages || [navigator.language])
    .map((tag) => String(tag || "").slice(0, 2).toLowerCase())
    .find((code) => code === "en" || code === "ru");
  return browser || "en";
}

let voiceLanguage = readVoiceLanguage();
const voiceLanguageSelects = document.querySelectorAll(".voice-language");
voiceLanguageSelects.forEach((select) => {
  select.value = voiceLanguage;
  select.addEventListener("change", () => {
    voiceLanguage = select.value;
    voiceLanguageSelects.forEach((other) => {
      other.value = voiceLanguage;
    });
    try {
      window.localStorage.setItem("voiceLanguage", voiceLanguage);
    } catch (error) {
      // the choice just won't survive a reload
    }
  });
});

// What the voice says. The default practice labels get a phrase; any other label is read as typed.
const voicePhrases = {
  en: {
    practices: { focus: "focus on the breath", open: "open awareness" },
    again: (phrase) => `${titleCase(phrase)} again.`,
    end: "That's it, you can open your eyes.",
    soundOn: "Sound on."
  },
  ru: {
    practices: { focus: "фокус на дыхании", open: "открытое внимание" },
    again: (phrase) => `Снова ${phrase}.`,
    end: "Всё, можно открыть глаза.",
    soundOn: "Звук включён."
  }
};
// The voices of the calibration run, preferred here too when the Mac has them.
const voiceNames = { en: "Samantha", ru: "Milena" };

function practicePhrase(condition) {
  const label = String(condition || "").trim();
  return voicePhrases[voiceLanguage].practices[label.toLowerCase()] || label;
}

// It speaks in the first seconds of a block, which the analysis drops anyway (the first 30 s of
// every block).
function phaseCue(phase, lastKey, blocks) {
  const phrases = voicePhrases[voiceLanguage];
  if (!phase.block) {
    return phrases.end;
  }
  const phrase = practicePhrase(phase.block.condition);
  const previous = blocks.find((block) => `block-${block.index}` === lastKey);
  return previous && previous.condition === phase.block.condition ? phrases.again(phrase) : `${titleCase(phrase)}.`;
}

// Chrome fills its voice list only after the first call, so ask now: the first cue then finds
// its voice instead of an empty list.
if (window.speechSynthesis) {
  window.speechSynthesis.getVoices();
}

// Only a voice of the Mac. A network one, like "Google русский" in Chrome, would send the text
// out of this machine, so without a local voice the change stays a tone. Samantha or Milena go
// first, then the system's default voice: the first English voice in the list can be a novelty one.
function localVoice(language) {
  const voices = window.speechSynthesis
    .getVoices()
    .filter((item) => item.localService && item.lang.toLowerCase().startsWith(language));
  return (
    voices.find((item) => item.name.startsWith(voiceNames[language])) ||
    voices.find((item) => item.default) ||
    voices[0]
  );
}

function speak(text, delayMs) {
  if (!window.speechSynthesis || typeof SpeechSynthesisUtterance === "undefined") {
    return;
  }
  window.setTimeout(() => {
    if (!chimeEnabled()) {
      return;
    }
    const voice = localVoice(voiceLanguage);
    if (!voice) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = voice.lang;
    utterance.voice = voice;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }, delayMs);
}

function planSeconds(recording, meditation) {
  const elapsed = numberOrNull(recording.elapsed_seconds);
  const offset = numberOrNull(meditation.first_frame_elapsed_seconds);
  if (elapsed == null || offset == null) {
    return null;
  }
  // progress.json moves every ~2 s; extrapolate in between so the countdown runs smoothly.
  const now = Date.now() / 1000;
  if (!meditationState.anchor || meditationState.anchor.value !== elapsed) {
    meditationState.anchor = { value: elapsed, at: now };
  }
  return meditationState.anchor.value + Math.min(3, now - meditationState.anchor.at) - offset;
}

function meditationPhase(blocks, seconds, active) {
  if (!active) {
    return { key: "done" };
  }
  if (seconds == null) {
    return { key: "waiting" };
  }
  if (seconds < blocks[0].start_s) {
    return { key: "settle", left: blocks[0].start_s - seconds };
  }
  for (const block of blocks) {
    if (seconds < block.end_s) {
      return { key: `block-${block.index}`, block, left: block.end_s - seconds };
    }
  }
  return { key: "done" };
}

function renderMeditation(recording) {
  const meditation = recording.meditation;
  const plan = meditation && meditation.plan;
  meditationPanel.hidden = !plan;
  if (!plan) {
    return;
  }
  if (meditationState.outputDir !== recording.output_dir) {
    meditationState.outputDir = recording.output_dir;
    meditationState.items = new Map();
    meditationState.lastPhase = null;
    meditationState.anchor = null;
    meditationBlocks.innerHTML = "";
  }
  const blocks = plan.blocks || [];
  const active = Boolean(recording.active);
  const seconds = planSeconds(recording, meditation);
  const phase = meditationPhase(blocks, seconds, active);
  meditationSound.hidden = !active;
  renderSoundButton();

  if (phase.key === "waiting") {
    meditationNow.textContent = "Waiting for the first headband data...";
  } else if (phase.key === "settle") {
    meditationNow.textContent = `Settling in · ${formatDuration(phase.left)}`;
  } else if (phase.block) {
    meditationNow.textContent = `Block ${phase.block.index + 1} of ${blocks.length} · ${phase.block.condition} · ${formatDuration(phase.left)} left`;
  } else {
    meditationNow.textContent = active
      ? "All blocks done. Rate them below; the recording is finishing."
      : "Done. Rate the blocks, then analyze.";
  }
  if (active && chimeEnabled() && !audioRunning()) {
    meditationNow.textContent += " (click the page to enable sound)";
  }

  // A cue when a block starts or the plan ends. A block right after "waiting" counts too: with
  // no settle-in the first block starts there.
  const last = meditationState.lastPhase;
  const blockStarted = Boolean(phase.block) && Boolean(last) && last !== phase.key;
  const planEnded = phase.key === "done" && Boolean(last) && last !== "done" && last !== "waiting";
  if ((blockStarted || planEnded) && active && chimeEnabled()) {
    const tones = phase.key === "done" ? 2 : 1;
    playChime(tones);
    speak(phaseCue(phase, last, blocks), 1400 + (tones - 1) * 900);
  }
  meditationState.lastPhase = phase.key;

  blocks.forEach((block) => renderMeditationBlock(block, seconds, active, phase));
  renderMeditationAnalysis(recording, meditation, active);
}

function soundBlocked() {
  return chimeEnabled() && !audioRunning() && !unlockTried;
}

function renderSoundButton() {
  const on = chimeEnabled();
  meditationSound.textContent = !on ? "Sound: off" : soundBlocked() ? "Enable sound" : "Sound: on";
  meditationSound.setAttribute("aria-pressed", String(on));
}

// Turning the sound on plays the tone and the voice once, so the volume can be checked before
// closing the eyes. While the browser blocks sound the button reads "Enable sound" and a click
// unlocks it instead of muting.
meditationSound.addEventListener("click", () => {
  if (chimeEnabled() && !soundBlocked()) {
    unlockTried = false;
    setChimeEnabled(false);
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  } else {
    unlockTried = chimeEnabled();
    setChimeEnabled(true);
    ensureAudio();
    playChime(1);
    speak(voicePhrases[voiceLanguage].soundOn, 1400);
  }
  renderSoundButton();
});

function renderMeditationBlock(block, seconds, active, phase) {
  let item = meditationState.items.get(block.index);
  if (!item) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    const ratings = document.createElement("span");
    ratings.className = "ratings";
    li.append(label, ratings);
    meditationBlocks.appendChild(li);
    item = { li, label, ratings, built: false };
    meditationState.items.set(block.index, item);
  }
  const done = !active || (seconds != null && seconds >= block.end_s);
  const current = phase.block && phase.block.index === block.index;
  item.li.className = current ? "current" : done ? "done" : "upcoming";
  item.label.textContent = `${block.condition} · ${(block.start_s / 60).toFixed(1)}–${(block.end_s / 60).toFixed(1)} min`;
  if (done && !item.built) {
    item.built = true;
    // Stability is stored as depth. Every save sends all three, since the server sets all three.
    const stability = ratingSelect("Stability", block.depth);
    const fading = ratingSelect("Sensory fading", block.sensory_fading);
    const drowsiness = ratingSelect("Drowsiness", block.drowsiness);
    const saved = document.createElement("small");
    const save = async () => {
      saved.textContent = "saving...";
      const response = await fetch("/api/meditation/rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          block_index: block.index,
          depth: stability.select.value,
          sensory_fading: fading.select.value,
          drowsiness: drowsiness.select.value
        })
      });
      saved.textContent = response.ok ? "saved" : "not saved";
    };
    [stability, fading, drowsiness].forEach((rating) => rating.select.addEventListener("change", save));
    item.ratings.append(stability.label, fading.label, drowsiness.label, saved);
  }
}

function ratingSelect(name, value) {
  const label = document.createElement("label");
  label.textContent = `${name} `;
  const select = document.createElement("select");
  ["", ...Array.from({ length: 11 }, (_, index) => String(index))].forEach((option) => {
    const element = document.createElement("option");
    element.value = option;
    element.textContent = option === "" ? "-" : option;
    select.appendChild(element);
  });
  select.value = value == null ? "" : String(Math.round(value));
  label.appendChild(select);
  return { label, select };
}

function renderMeditationAnalysis(recording, meditation, active) {
  const analysis = meditation.analysis || { state: "none" };
  const finished = !active && recording.state === "completed";
  analyzeMeditationButton.hidden = !finished || analysis.state === "ready";
  analyzeMeditationButton.disabled = analysis.state === "running";
  analyzeMeditationButton.textContent = analysis.state === "failed" ? "Analyze again" : "Analyze meditation";
  openMeditationReport.hidden = !(analysis.state === "ready" && analysis.url);
  if (analysis.url) {
    openMeditationReport.href = analysis.url;
  }
  meditationAnalysisText.textContent =
    analysis.state === "running"
      ? "Analyzing (about a minute)..."
      : analysis.state === "failed"
      ? `Analysis failed, see ${analysis.log_path}`
      : "";
}

// The forms open under all the mode cards, so one may open below the window's edge.
function revealForm(form) {
  if (!form.hidden) {
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    form.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
  }
}

startMeditationButton.addEventListener("click", () => {
  meditationForm.hidden = !meditationForm.hidden;
  meditationChime.checked = chimeEnabled();
  revealForm(meditationForm);
});

// Cancelling a form also drops a start that is still waiting for contact.
meditationCancel.addEventListener("click", () => {
  meditationForm.hidden = true;
  cancelPendingStart();
});

meditationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setChimeEnabled(meditationChime.checked);
  if (meditationChime.checked) {
    ensureAudio();
  }
  const body = {
    conditions: [document.querySelector("#med-condition-a").value, document.querySelector("#med-condition-b").value],
    blocks: Number(document.querySelector("#med-blocks").value),
    block_minutes: Number(document.querySelector("#med-block-minutes").value),
    settle_seconds: Number(document.querySelector("#med-settle").value),
    with_polar: withPolarCheckbox.checked
  };
  startWhenReady({
    label: "meditation",
    card: "meditation",
    url: "/api/meditation/start",
    body,
    done: (response, payload) => {
      const formError = document.querySelector("#med-error");
      if (response.ok) {
        meditationForm.hidden = true;
        formError.hidden = true;
      } else {
        formError.hidden = false;
        formError.textContent = payload.error || `${response.status} ${response.statusText}`;
      }
    }
  });
});

analyzeMeditationButton.addEventListener("click", async () => {
  analyzeMeditationButton.disabled = true;
  try {
    await requestJson("/api/meditation/analyze", { method: "POST" });
  } finally {
    await refreshUiState();
  }
});

// --- meditation series --------------------------------------------------------

const seriesCount = document.querySelector("#series-count");
const seriesTarget = document.querySelector("#series-target");
const seriesSessions = document.querySelector("#series-sessions");
const seriesStart = document.querySelector("#series-start");
const seriesNew = document.querySelector("#series-new");
const seriesChime = document.querySelector("#series-chime");
const seriesError = document.querySelector("#series-error");
const seriesConditionA = document.querySelector("#series-condition-a");
const seriesConditionB = document.querySelector("#series-condition-b");
const seriesBlocks = document.querySelector("#series-blocks");
const seriesBlockMinutes = document.querySelector("#series-block-minutes");
const seriesSettle = document.querySelector("#series-settle");
const seriesDots = document.querySelector("#series-dots");
const seriesCardTarget = document.querySelector("#series-card-target");
let seriesStatus = null;
let seriesSawRecording = false;

async function loadSeries() {
  try {
    seriesStatus = await requestJson("/api/meditation/series");
  } catch (error) {
    seriesStatus = null;
  }
  renderSeries();
}

// The count changes when a recording ends.
function reloadSeriesAfterRecording(recording) {
  const active = Boolean(recording.active);
  if (seriesSawRecording && !active) {
    loadSeries();
  }
  seriesSawRecording = active;
}

function renderSeries() {
  const status = seriesStatus || {};
  const series = status.series;
  const counted = status.counted || 0;
  const target = status.target || 8;
  startSeriesButton.textContent = series ? `Session ${counted + 1}…` : "Set up…";
  seriesTarget.textContent = String(target);
  seriesCardTarget.textContent = String(target);
  renderSeriesDots(series, counted, target);
  seriesCount.textContent = !series
    ? "no sessions yet"
    : counted < target
    ? `${counted} of ${target} sessions`
    : `${counted} sessions, enough for p-values`;
  // The first session sets the practices and the block layout; later ones keep them.
  [seriesConditionA, seriesConditionB, seriesBlocks, seriesBlockMinutes].forEach((input) => {
    input.disabled = Boolean(series);
  });
  if (series) {
    seriesConditionA.value = series.conditions[0];
    seriesConditionB.value = series.conditions[1];
    seriesBlocks.value = series.blocks;
    seriesBlockMinutes.value = series.block_minutes;
  }
  const sessions = status.sessions || [];
  seriesSessions.innerHTML = "";
  seriesSessions.hidden = sessions.length === 0;
  let number = 0;
  sessions.forEach((session) => {
    const item = document.createElement("li");
    const started = formatStart(session.started_at, session.name);
    if (session.state === "counted") {
      number += 1;
      item.className = "done";
      item.textContent = `Session ${number} · ${started}`;
    } else {
      item.className = "upcoming";
      item.textContent = `${started} · ${seriesSessionText(session)}`;
    }
    seriesSessions.appendChild(item);
  });
  seriesStart.textContent = `Start session ${counted + 1}`;
  seriesNew.hidden = !series;
}

function renderSeriesDots(series, counted, target) {
  seriesDots.innerHTML = "";
  if (!series) {
    seriesDots.textContent = "No series yet";
    return;
  }
  for (let index = 0; index < target; index += 1) {
    const dot = document.createElement("i");
    dot.className = index < counted ? "on" : "";
    seriesDots.appendChild(dot);
  }
  const label = document.createElement("span");
  label.textContent = counted < target ? `${counted} of ${target}` : `${counted} counted`;
  seriesDots.appendChild(label);
  seriesDots.setAttribute("aria-label", `${counted} of ${target} sessions counted`);
}

function seriesSessionText(session) {
  if (session.state === "recording") {
    return "recording now";
  }
  if (session.state === "short") {
    return `headband data for ${minutesClock(session.covered_seconds)} of ${minutesClock(session.needed_seconds)}, not counted`;
  }
  if (session.state === "no_data") {
    return "no headband data, not counted";
  }
  return "the recorder did not finish, not counted";
}

startSeriesButton.addEventListener("click", async () => {
  seriesForm.hidden = !seriesForm.hidden;
  revealForm(seriesForm);
  if (seriesForm.hidden) {
    return;
  }
  seriesChime.checked = chimeEnabled();
  seriesError.hidden = true;
  await loadSeries();
  const series = seriesStatus && seriesStatus.series;
  if (series && series.settle_seconds != null) {
    seriesSettle.value = series.settle_seconds;
  }
});

document.querySelector("#series-cancel").addEventListener("click", () => {
  seriesForm.hidden = true;
  cancelPendingStart();
});

seriesNew.addEventListener("click", async () => {
  const counted = (seriesStatus && seriesStatus.counted) || 0;
  if (!window.confirm(`Start a new series? The ${counted} counted session(s) stay on disk but stop counting here.`)) {
    return;
  }
  try {
    seriesStatus = await requestJson("/api/meditation/series/new", { method: "POST" });
  } catch (error) {
    seriesError.hidden = false;
    seriesError.textContent = `Could not start a new series: ${error.message}`;
  }
  renderSeries();
});

seriesForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setChimeEnabled(seriesChime.checked);
  if (seriesChime.checked) {
    ensureAudio();
  }
  startWhenReady({
    label: "series session",
    card: "series",
    url: "/api/meditation/series/start",
    body: {
      conditions: [seriesConditionA.value, seriesConditionB.value],
      blocks: Number(seriesBlocks.value),
      block_minutes: Number(seriesBlockMinutes.value),
      settle_seconds: Number(seriesSettle.value)
    },
    done: async (response, payload) => {
      if (response.ok) {
        seriesForm.hidden = true;
        seriesError.hidden = true;
      } else {
        seriesError.hidden = false;
        seriesError.textContent = payload.error || `${response.status} ${response.statusText}`;
      }
      await loadSeries();
    }
  });
});

// --- calibration run -----------------------------------------------------------

function minutesClock(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(value / 60)}:${pad(value % 60)}`;
}

function renderCalibration(recording) {
  const calibration = recording.calibration;
  calibrationPanel.hidden = !calibration;
  if (!calibration) {
    return;
  }
  const active = Boolean(recording.active);
  const step = calibration.step || {};
  if (active) {
    if (step.state === "running" && step.label) {
      calibrationNow.textContent = `Step ${step.step} of ${step.steps} · ${step.label} · ${minutesClock(step.start_s)}–${minutesClock(step.end_s)}`;
    } else if (step.state === "done") {
      calibrationNow.textContent = "All steps done; the recording is finishing.";
    } else {
      calibrationNow.textContent = "Waiting for the first headband data, then the voice starts.";
    }
    if (!calibration.guide_running && step.state !== "done") {
      calibrationNow.textContent += ` The voice guide is not running, see ${calibration.guide_log}.`;
    }
  } else {
    calibrationNow.textContent =
      step.state === "done" ? "Calibration run finished." : `Calibration run stopped at step ${step.step || "?"}.`;
  }
  const report = calibration.report || { state: "none" };
  const finished = !active && recording.state === "completed";
  calibrationReportButton.hidden = !finished || report.state === "ready";
  calibrationReportButton.disabled = report.state === "running";
  calibrationReportButton.textContent = report.state === "failed" ? "Build again" : "Build calibration report";
  openCalibrationReport.hidden = !(report.state === "ready" && report.url);
  if (report.url) {
    openCalibrationReport.href = report.url;
  }
  calibrationReportText.textContent =
    report.state === "running"
      ? "Building (a minute or two)..."
      : report.state === "failed"
      ? `Report failed, see ${report.log_path}`
      : "";
}

startCalibrationButton.addEventListener("click", async () => {
  calibrationForm.hidden = !calibrationForm.hidden;
  revealForm(calibrationForm);
  if (calibrationForm.hidden || calibrationSteps.childElementCount) {
    return;
  }
  try {
    const protocol = await requestJson("/api/calibration/protocol");
    (protocol.steps || []).forEach((step) => {
      const item = document.createElement("li");
      item.className = "upcoming";
      item.textContent = `${minutesClock(step.start_s)}–${minutesClock(step.end_s)} · ${step.label}`;
      calibrationSteps.appendChild(item);
    });
  } catch (error) {
    calibrationSteps.textContent = `Could not load the steps: ${error.message}`;
  }
});

document.querySelector("#calibration-cancel").addEventListener("click", () => {
  calibrationForm.hidden = true;
  cancelPendingStart();
});

calibrationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  startWhenReady({
    label: "calibration run",
    card: "calibration",
    url: "/api/calibration/start",
    body: { language: voiceLanguage },
    done: (response, payload) => {
      const formError = document.querySelector("#calibration-error");
      if (response.ok) {
        calibrationForm.hidden = true;
        formError.hidden = true;
      } else {
        formError.hidden = false;
        formError.textContent = payload.error || `${response.status} ${response.statusText}`;
      }
    }
  });
});

calibrationReportButton.addEventListener("click", async () => {
  calibrationReportButton.disabled = true;
  try {
    await requestJson("/api/calibration/report", { method: "POST" });
  } finally {
    await refreshUiState();
  }
});

// --- tabs and the recordings list ------------------------------------------------

const historyList = document.querySelector("#history-list");
const historyCount = document.querySelector("#history-count");
const historyError = document.querySelector("#history-error");
const recordingFilters = document.querySelector("#recording-filters");
let historyRecordings = [];
let historyTotal = 0;
let historyFilter = "all";
// "Show older" asks for more; the list only redraws when what the server sends changes.
const HISTORY_PAGE = 30;
let historyLimit = HISTORY_PAGE;
let historyJson = "";

function showScreen(name) {
  const screens = {
    record: [tabRecord, screenRecord],
    morning: [tabMorning, screenMorning],
    recordings: [tabRecordings, screenRecordings]
  };
  const shown = screens[name] ? name : "record";
  Object.entries(screens).forEach(([key, [tab, screen]]) => {
    screen.hidden = key !== shown;
    if (key === shown) {
      tab.setAttribute("aria-current", "page");
    } else {
      tab.removeAttribute("aria-current");
    }
  });
  try {
    window.history.replaceState(null, "", shown === "record" ? window.location.pathname : `#${shown}`);
  } catch (error) {
    // the tab just won't survive a reload
  }
  if (shown === "recordings") {
    loadHistory();
  }
  if (shown === "morning") {
    // Coming back to the tab keeps what is typed in the forms; only the list of nights refreshes.
    loadMorning({ keepForms: morningNight != null });
  }
}

tabRecord.addEventListener("click", () => showScreen("record"));
tabRecordings.addEventListener("click", () => showScreen("recordings"));

recordingFilters.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    historyFilter = chip.dataset.filter;
    recordingFilters.querySelectorAll(".chip").forEach((other) => {
      other.setAttribute("aria-pressed", String(other === chip));
    });
    renderHistory(historyRecordings, historyTotal);
  });
});

async function loadHistory() {
  if (screenRecordings.hidden) {
    return;
  }
  try {
    const payload = await requestJson(`/api/recordings?limit=${historyLimit}`);
    const json = JSON.stringify(payload);
    if (json === historyJson) {
      return;
    }
    historyJson = json;
    renderHistory(payload.recordings || [], payload.total || 0);
  } catch (error) {
    historyJson = "";
    historyList.textContent = `Could not list recordings: ${error.message}`;
  }
}

function recordingCategory(recording) {
  if (recording.calibration) {
    return "calibration";
  }
  if (recording.meditation) {
    return "meditation";
  }
  return recording.kind === "night" ? "night" : "session";
}

function renderHistory(recordings, total) {
  historyRecordings = recordings;
  historyTotal = Math.max(total, recordings.length);
  const shown = recordings.filter((recording) => historyFilter === "all" || recordingCategory(recording) === historyFilter);
  historyCount.textContent = historyTotal === 1 ? "1 recording" : `${historyTotal} recordings`;
  historyList.innerHTML = "";
  if (shown.length === 0) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = recordings.length === 0 ? "No recordings yet." : "No recordings of this kind among these.";
    historyList.appendChild(item);
  } else {
    shown.forEach((recording) => historyList.appendChild(historyRow(recording)));
  }
  const older = historyTotal - recordings.length;
  if (older > 0) {
    const item = document.createElement("li");
    item.className = "more";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Show older (${older} more)`;
    button.addEventListener("click", () => {
      historyLimit += HISTORY_PAGE;
      loadHistory();
    });
    item.appendChild(button);
    historyList.appendChild(item);
  }
}

function historyRow(recording) {
  const item = document.createElement("li");

  const when = document.createElement("div");
  when.className = "rec-when";
  const started = recording.started_at ? new Date(recording.started_at) : null;
  when.textContent = started
    ? started.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })
    : recording.name;
  const time = document.createElement("small");
  time.textContent = [
    started ? started.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null,
    formatMinutes(recording.duration_seconds)
  ]
    .filter(Boolean)
    .join(" · ");
  when.appendChild(time);

  const what = document.createElement("div");
  what.className = "rec-what";
  const category = recordingCategory(recording);
  const chip = document.createElement("span");
  chip.className = `kind-chip ${recording.live ? "live" : category}`;
  chip.textContent = recording.live ? "Recording now" : titleCase(category);
  const detail = document.createElement("span");
  detail.textContent = [
    recording.live ? null : recording.stop_reason ? String(recording.stop_reason).replaceAll("_", " ") : "no summary",
    recording.with_polar ? "H10" : null
  ]
    .filter(Boolean)
    .join(" · ");
  what.append(chip, detail);

  const signal = document.createElement("div");
  signal.className = "rec-signal";
  signal.textContent = signalText(recording.signal);

  const actions = document.createElement("div");
  actions.className = "rec-actions";
  if (!recording.live) {
    actions.append(jobControl(recording, recording.report, "REM report", "Build REM report", "/api/recordings/report"));
    if (recording.meditation) {
      actions.append(
        jobControl(recording, recording.meditation_report, "Meditation report", "Analyze", "/api/recordings/analyze")
      );
    }
    if (recording.calibration) {
      actions.append(
        jobControl(
          recording,
          recording.calibration_report,
          "Calibration report",
          "Build calibration report",
          "/api/recordings/calibration-report"
        )
      );
    }
  }
  item.append(when, what, signal, actions);
  return item;
}

function signalText(signal) {
  if (!signal) {
    return "";
  }
  const rate = numberOrNull(signal.eeg_rate_hz);
  const lost = numberOrNull(signal.lost_subpackets);
  return [rate == null ? null : `EEG ${rate.toFixed(1)} Hz`, lost == null ? null : `${lost} lost`]
    .filter(Boolean)
    .join(" · ");
}

function jobControl(recording, status, openLabel, buildLabel, endpoint) {
  const state = (status && status.state) || "none";
  if (state === "ready" && status.url) {
    const link = document.createElement("a");
    link.href = status.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = openLabel;
    return link;
  }
  const button = document.createElement("button");
  button.type = "button";
  button.disabled = state === "running";
  button.textContent = state === "running" ? "Working..." : state === "failed" ? `${buildLabel} (retry)` : buildLabel;
  button.title = state === "failed" ? `Failed, see ${status.log_path}` : "";
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Working...";
    historyError.hidden = true;
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: recording.kind, name: recording.name })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `${response.status} ${response.statusText}`);
      }
    } catch (error) {
      historyError.hidden = false;
      historyError.textContent = `${buildLabel} for ${recording.name}: ${error.message}`;
    } finally {
      // Draw the list from what the server says now; a failed launch leaves it unchanged.
      historyJson = "";
      await loadHistory();
    }
  });
  return button;
}

function formatStart(startedAt, fallback) {
  if (!startedAt) {
    return fallback;
  }
  const date = new Date(startedAt);
  return date.toLocaleString([], { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// --- the morning after a puzzle night --------------------------------------------
// The dream report, the sound check and the blind retest save through the same code as
// the CLI. The server never sends which puzzles were cued or which arm the night had, so
// the page cannot give either away: the sound check knows its sounds only by number.

const tabMorning = document.querySelector("#tab-morning");
const screenMorning = document.querySelector("#screen-morning");
const morningNightSelect = document.querySelector("#morning-night");
const morningEmpty = document.querySelector("#morning-empty");
const morningWarning = document.querySelector("#morning-warning");
const morningForms = document.querySelector("#morning-forms");
const dreamForm = document.querySelector("#dream-form");
const dreamStatus = document.querySelector("#dream-status");
const dreamConfidence = document.querySelector("#dream-confidence");
const dreamText = document.querySelector("#dream-text");
const dreamLinks = document.querySelector("#dream-links");
const dreamNotes = document.querySelector("#dream-notes");
const dreamEyeSignals = document.querySelector("#dream-eye-signals");
const dreamError = document.querySelector("#dream-error");
const recallForm = document.querySelector("#recall-form");
const recallStatus = document.querySelector("#recall-status");
const recallLate = document.querySelector("#recall-late");
const recallList = document.querySelector("#recall-list");
const recallNotes = document.querySelector("#recall-notes");
const recallError = document.querySelector("#recall-error");
const recallSave = document.querySelector("#recall-save");
const retestForm = document.querySelector("#retest-form");
const retestStatus = document.querySelector("#retest-status");
const retestWave = document.querySelector("#retest-wave");
const retestList = document.querySelector("#retest-list");
const retestNotes = document.querySelector("#retest-notes");
const retestError = document.querySelector("#retest-error");
const retestSave = document.querySelector("#retest-save");
let morningNight = null;
// Per puzzle: the seconds spent so far and when its running timer started.
const retestTimers = new Map();
// Typed since the form was last drawn from what is saved; an app update waits for them.
let dreamUnsaved = false;
let recallUnsaved = false;
let retestUnsaved = false;
// Per sound number: how often it was played, and when the check began.
const recallPlays = new Map();
let recallStartedAt = null;

dreamForm.addEventListener("input", () => {
  dreamUnsaved = true;
});
recallForm.addEventListener("input", () => {
  recallUnsaved = true;
});
retestForm.addEventListener("input", () => {
  retestUnsaved = true;
});

function morningUnsaved() {
  return dreamUnsaved || recallUnsaved || retestUnsaved;
}

tabMorning.addEventListener("click", () => showScreen("morning"));

async function loadMorning({ keepForms = false } = {}) {
  let nights = [];
  try {
    nights = (await requestJson("/api/morning/nights")).nights || [];
  } catch (error) {
    showMorningWarning(`Could not list the puzzle nights: ${error.message}`);
    return;
  }
  const current = morningNightSelect.value;
  morningNightSelect.innerHTML = "";
  nights.forEach((night) => {
    const option = document.createElement("option");
    option.value = night.name;
    const done = night.dream_report_saved && night.retest_saved ? " · done" : "";
    option.textContent = `${night.name} · ${night.puzzle_count} puzzles${done}`;
    morningNightSelect.appendChild(option);
  });
  morningEmpty.hidden = nights.length > 0;
  morningNightSelect.parentElement.hidden = nights.length === 0;
  morningForms.hidden = nights.length === 0;
  if (nights.length === 0) {
    showMorningWarning("");
    return;
  }
  const stillThere = nights.some((night) => night.name === current);
  morningNightSelect.value = stillThere ? current : nights[0].name;
  if (!keepForms || !stillThere) {
    await loadMorningNight(morningNightSelect.value);
  }
}

morningNightSelect.addEventListener("change", () => loadMorningNight(morningNightSelect.value));

async function loadMorningNight(name) {
  stopRetestTimers();
  let night;
  try {
    night = await requestJson(`/api/morning/night?name=${encodeURIComponent(name)}`);
  } catch (error) {
    showMorningWarning(`Could not load ${name}: ${error.message}`);
    return;
  }
  // Another night was picked while this one loaded; that load draws the forms.
  if (morningNightSelect.value !== name) {
    return;
  }
  morningNight = night;
  renderDreamForm(morningNight);
  renderRecallForm(morningNight);
  renderRetestForm(morningNight);
  const missing = [];
  if (!morningNight.catalog_file) {
    missing.push("no puzzle catalog in data/protocol holds this night's puzzles, so their text is missing");
  }
  if (!morningNight.assignment_found) {
    missing.push("its cue assignment is missing");
  }
  showMorningWarning(missing.length ? `The retest cannot be saved: ${missing.join(", and ")}.` : "");
}

function showMorningWarning(text) {
  morningWarning.hidden = !text;
  morningWarning.textContent = text;
}

function savedText(saved) {
  if (!saved) {
    return "Not saved yet.";
  }
  const at = saved.reported_at_utc ? new Date(saved.reported_at_utc) : null;
  const when = at && !Number.isNaN(at.getTime())
    ? at.toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })
    : "earlier";
  return `Saved ${when}. Saving again replaces it.`;
}

function confidenceOptions(select, value) {
  select.innerHTML = "";
  ["", ...Array.from({ length: 11 }, (_, index) => String(index))].forEach((option) => {
    const element = document.createElement("option");
    element.value = option;
    element.textContent = option === "" ? "-" : option;
    select.appendChild(element);
  });
  select.value = value == null ? "" : String(Math.round(value * 10));
}

function renderDreamForm(night) {
  const saved = night.dream_report;
  dreamStatus.textContent = savedText(saved);
  setRadio("dream-lucid", saved ? (saved.lucid ? "yes" : "no") : "");
  setRadio("dream-perception", saved ? saved.cue_perception : "");
  setRadio("dream-woke", saved ? saved.woke_from_sound : "");
  confidenceOptions(dreamConfidence, saved ? saved.confidence : null);
  dreamText.value = saved ? saved.dream_text : "";
  dreamNotes.value = saved ? saved.notes : "";
  dreamEyeSignals.value = saved ? (saved.eye_signal_attempt_times || []).join(", ") : "";
  dreamLinks.innerHTML = "";
  const intro = document.createElement("p");
  intro.className = "muted";
  intro.textContent = "If the dream touched one of last night's puzzles, say how:";
  dreamLinks.appendChild(intro);
  night.puzzles.forEach((puzzle) => {
    const label = document.createElement("label");
    label.className = "field";
    const title = document.createElement("span");
    title.textContent = `Puzzle ${puzzle.position}: ${puzzle.prompt || puzzle.puzzle_id}`;
    const area = document.createElement("textarea");
    area.rows = 2;
    area.dataset.puzzleId = puzzle.puzzle_id;
    area.value = (saved && saved.puzzle_links[puzzle.puzzle_id]) || "";
    label.append(title, area);
    dreamLinks.appendChild(label);
  });
  dreamError.hidden = true;
  dreamUnsaved = false;
}

function setRadio(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    input.checked = input.value === value;
  });
}

function radioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : "";
}

dreamForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const links = {};
  dreamLinks.querySelectorAll("textarea").forEach((area) => {
    if (area.value.trim()) {
      links[area.dataset.puzzleId] = area.value.trim();
    }
  });
  await saveMorningForm(
    "/api/morning/dream-report",
    {
      night: morningNight.name,
      lucid: radioValue("dream-lucid"),
      cue_perception: radioValue("dream-perception"),
      woke_from_sound: radioValue("dream-woke"),
      eye_signal_attempt_times: dreamEyeSignals.value,
      confidence: dreamConfidence.value === "" ? "" : Number(dreamConfidence.value) / 10,
      dream_text: dreamText.value,
      puzzle_links: links,
      notes: dreamNotes.value
    },
    dreamError,
    "dream report",
    renderDreamForm
  );
});

function renderRecallForm(night) {
  const recall = night.cue_recall || { trials: [], available: false, unavailable_reason: null, saved: null };
  const saved = recall.saved;
  recallStatus.textContent = recall.available || saved
    ? savedText(saved)
    : `The sounds cannot play: ${recall.unavailable_reason}.`;
  recallLate.hidden = !(night.retest && !saved);
  recallNotes.value = saved ? saved.notes : "";
  recallList.innerHTML = "";
  recallPlays.clear();
  recallStartedAt = null;
  recall.trials.forEach((trial) => {
    recallList.appendChild(recallItem(night, trial.trial, saved ? saved.answers[String(trial.trial)] : undefined, recall.available));
  });
  recallError.hidden = true;
  recallSave.disabled = !recall.available;
  recallUnsaved = false;
}

function recallItem(night, number, saved, playable) {
  const item = document.createElement("li");
  item.className = "recall-item";
  item.dataset.trial = String(number);
  const title = document.createElement("b");
  title.textContent = `Sound ${number}`;
  const play = document.createElement("button");
  play.type = "button";
  play.textContent = "Play";
  play.disabled = !playable;
  play.addEventListener("click", () => playRecallSound(night.name, number, play));
  const choice = document.createElement("select");
  choice.setAttribute("aria-label", `Puzzle for sound ${number}`);
  const options = [["", "Pick the puzzle"]];
  night.puzzles.forEach((puzzle) => {
    options.push([puzzle.puzzle_id, `Puzzle ${puzzle.position}: ${puzzle.prompt || puzzle.puzzle_id}`]);
  });
  options.push(["unsure", "Not sure"]);
  options.forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    choice.appendChild(option);
  });
  choice.value = saved === undefined ? "" : saved === null ? "unsure" : saved;
  item.append(title, play, choice);
  return item;
}

async function playRecallSound(night, number, button) {
  recallError.hidden = true;
  button.disabled = true;
  if (recallStartedAt == null) {
    recallStartedAt = new Date().toISOString();
  }
  try {
    const response = await postJson("/api/morning/cue-recall/play", { night, trial: number });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      recallError.hidden = false;
      recallError.textContent = payload.error || `${response.status} ${response.statusText}`;
      return;
    }
    recallPlays.set(number, (recallPlays.get(number) || 0) + 1);
    recallUnsaved = true;
  } catch (error) {
    recallError.hidden = false;
    recallError.textContent = `Could not play sound ${number}: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

recallForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const answers = {};
  let incomplete = false;
  recallList.querySelectorAll(".recall-item").forEach((item) => {
    const value = item.querySelector("select").value;
    if (value === "") {
      incomplete = true;
    }
    answers[item.dataset.trial] = value === "unsure" ? null : value;
  });
  if (incomplete) {
    recallError.hidden = false;
    recallError.textContent = "Every sound needs a puzzle, or Not sure.";
    return;
  }
  await saveMorningForm(
    "/api/morning/cue-recall",
    {
      night: morningNight.name,
      answers,
      play_counts: Object.fromEntries(recallPlays),
      started_at_utc: recallStartedAt || "",
      notes: recallNotes.value
    },
    recallError,
    "sound check",
    renderRecallForm
  );
});

// The retest of the wave picked: this morning's, a week's or ten days' later.
function savedRetest(night, wave) {
  if (night.retests && wave in night.retests) {
    return night.retests[wave];
  }
  return wave === "morning" ? night.retest : null;
}

function renderRetestForm(night) {
  const wave = retestWave.value || "morning";
  retestWave.dataset.shown = wave;
  const saved = savedRetest(night, wave);
  retestStatus.textContent = savedText(saved);
  retestNotes.value = saved ? saved.notes : "";
  retestTimers.clear();
  retestList.innerHTML = "";
  night.puzzles.forEach((puzzle) => {
    retestList.appendChild(retestItem(puzzle, saved ? saved.results[puzzle.puzzle_id] : null));
  });
  retestError.hidden = true;
  retestSave.disabled = !night.catalog_file || !night.assignment_found;
  retestUnsaved = false;
}

function retestItem(puzzle, saved) {
  const item = document.createElement("li");
  item.className = "retest-item";
  item.dataset.puzzleId = puzzle.puzzle_id;

  const head = document.createElement("div");
  head.className = "retest-head";
  const title = document.createElement("b");
  title.textContent = `Puzzle ${puzzle.position}`;
  const elapsed = document.createElement("span");
  elapsed.className = "elapsed";
  const timer = document.createElement("button");
  timer.type = "button";
  head.append(title, elapsed, timer);

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = puzzle.prompt || `(${puzzle.puzzle_id}: not in any catalog)`;

  const answerLabel = document.createElement("label");
  answerLabel.className = "field";
  answerLabel.textContent = "Your answer";
  const answer = document.createElement("textarea");
  answer.rows = 3;
  answer.value = saved ? saved.response : "";
  answerLabel.appendChild(answer);

  const row = document.createElement("div");
  row.className = "retest-row";
  const solvedLabel = document.createElement("label");
  const solved = document.createElement("input");
  solved.type = "checkbox";
  solved.checked = Boolean(saved && saved.solved);
  solvedLabel.append(solved, "Solved");
  const confidenceLabel = document.createElement("label");
  confidenceLabel.textContent = "Confidence (0-10) ";
  const confidence = document.createElement("select");
  confidenceOptions(confidence, saved ? saved.confidence : null);
  confidenceLabel.appendChild(confidence);
  const secondsLabel = document.createElement("label");
  secondsLabel.textContent = "Seconds ";
  const seconds = document.createElement("input");
  seconds.type = "number";
  seconds.min = "0";
  seconds.step = "1";
  seconds.value = saved ? String(Math.round(saved.duration_seconds)) : "";
  secondsLabel.appendChild(seconds);
  row.append(solvedLabel, confidenceLabel, secondsLabel);

  const noteLabel = document.createElement("label");
  noteLabel.className = "field";
  noteLabel.textContent = "Note";
  const note = document.createElement("input");
  note.className = "retest-note";
  note.maxLength = 500;
  note.value = saved ? saved.notes : "";
  noteLabel.appendChild(note);

  item.append(head, prompt, answerLabel, row, noteLabel);

  const state = { item, timer, elapsed, seconds, spent: saved ? Number(saved.duration_seconds) || 0 : 0, startedAt: null };
  retestTimers.set(puzzle.puzzle_id, state);
  timer.addEventListener("click", () => toggleRetestTimer(state));
  // A typed total replaces the time so far; a running timer goes on counting from it.
  seconds.addEventListener("change", () => {
    state.spent = Number(seconds.value) || 0;
    if (state.startedAt != null) {
      state.startedAt = Date.now();
    }
    renderRetestTimer(state);
  });
  renderRetestTimer(state);
  return item;
}

function retestSeconds(state) {
  return state.spent + (state.startedAt == null ? 0 : (Date.now() - state.startedAt) / 1000);
}

function renderRetestTimer(state) {
  const running = state.startedAt != null;
  state.timer.textContent = running ? "Done" : state.spent > 0 ? "Resume" : "Start";
  state.elapsed.textContent = formatDuration(retestSeconds(state));
  state.item.classList.toggle("timing", running);
  // Not while the field is being typed into.
  if (running && document.activeElement !== state.seconds) {
    state.seconds.value = String(Math.round(retestSeconds(state)));
  }
}

// One puzzle at a time: starting a timer stops the one that was running.
function toggleRetestTimer(state) {
  retestUnsaved = true;
  const running = state.startedAt != null;
  stopRetestTimers();
  if (!running) {
    state.startedAt = Date.now();
  }
  renderRetestTimer(state);
}

function stopRetestTimers() {
  retestTimers.forEach((state) => {
    if (state.startedAt != null) {
      state.spent = retestSeconds(state);
      state.startedAt = null;
      state.seconds.value = String(Math.round(state.spent));
      renderRetestTimer(state);
    }
  });
}

window.setInterval(() => {
  retestTimers.forEach((state) => {
    if (state.startedAt != null) {
      renderRetestTimer(state);
    }
  });
}, 1000);

retestWave.addEventListener("change", () => {
  if (retestUnsaved && !window.confirm("Leave the retest answers that are not saved?")) {
    retestWave.value = retestWave.dataset.shown || "morning";
    return;
  }
  stopRetestTimers();
  if (morningNight) {
    renderRetestForm(morningNight);
  }
});

retestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopRetestTimers();
  const results = [];
  let incomplete = false;
  retestTimers.forEach((state, puzzleId) => {
    const confidence = state.item.querySelector("select").value;
    if (state.seconds.value === "" || confidence === "") {
      incomplete = true;
    }
    results.push({
      puzzle_id: puzzleId,
      response: state.item.querySelector("textarea").value,
      solved: state.item.querySelector('input[type="checkbox"]').checked,
      duration_seconds: Number(state.seconds.value),
      confidence: confidence === "" ? "" : Number(confidence) / 10,
      notes: state.item.querySelector(".retest-note").value
    });
  });
  if (incomplete) {
    retestError.hidden = false;
    retestError.textContent = "Every puzzle needs its time and a confidence, even with no answer.";
    return;
  }
  await saveMorningForm(
    "/api/morning/retest",
    { night: morningNight.name, wave: retestWave.value || "morning", results, notes: retestNotes.value },
    retestError,
    "retest",
    renderRetestForm
  );
});

function postJson(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

// After a save only that form is drawn again: the other one may hold answers not saved yet.
async function saveMorningForm(url, body, errorBox, what, render) {
  errorBox.hidden = true;
  // The night stays put while it saves, so the answers cannot end up under another one.
  morningNightSelect.disabled = true;
  try {
    let response = await postJson(url, body);
    let payload = await response.json().catch(() => ({}));
    if (response.status === 409 && payload.exists) {
      if (!window.confirm(`Replace the saved ${what}?`)) {
        return;
      }
      response = await postJson(url, { ...body, replace: true });
      payload = await response.json().catch(() => ({}));
    }
    if (!response.ok) {
      errorBox.hidden = false;
      errorBox.textContent = payload.error || `${response.status} ${response.statusText}`;
      return;
    }
    try {
      const night = await requestJson(`/api/morning/night?name=${encodeURIComponent(body.night)}`);
      if (morningNightSelect.value === body.night) {
        morningNight = night;
        render(morningNight);
      }
    } catch (error) {
      showMorningWarning(`Saved, but could not reload ${body.night}: ${error.message}`);
    }
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = `Could not save the ${what}: ${error.message}`;
  } finally {
    morningNightSelect.disabled = false;
  }
  await loadMorning({ keepForms: true });
}

window.setInterval(loadHistory, 5000);
showScreen(window.location.hash.slice(1) || "record");

refreshUiState();
window.setInterval(() => {
  // Checked here too: a poll that hangs never comes back to say so.
  renderAppLost();
  refreshUiState();
}, 1000);
loadSeries();

function updateContactHistory(channel, fill) {
  const now = Date.now() / 1000;
  const history = contactHistory[channel];
  history.push({ time: now, fill });
  while (history.length > 0 && (now - history[0].time > 30 || history.length > 40)) {
    history.shift();
  }
}

function sparklinePoints(history) {
  if (!history || history.length === 0) {
    return "0,20 56,20";
  }
  if (history.length === 1) {
    const y = sparklineY(history[0].fill);
    return `0,${y} 56,${y}`;
  }
  const last = history[history.length - 1].time;
  const first = Math.max(last - 30, history[0].time);
  const span = Math.max(1, last - first);
  return history
    .map((point) => {
      const x = clamp((point.time - first) / span, 0, 1) * 56;
      return `${x.toFixed(1)},${sparklineY(point.fill)}`;
    })
    .join(" ");
}

function sparklineY(fill) {
  return (20 - clamp(fill, 0, 1) * 18).toFixed(1);
}

function primaryReason(reasonCodes) {
  if (!reasonCodes || reasonCodes.length === 0) {
    return "";
  }
  return reasonText[reasonCodes[0]] || reasonCodes[0].replaceAll("_", " ");
}

function warningEventText(event) {
  const time = formatClock(event.timestamp_seconds);
  if (event.kind === "contact_recovered") {
    return `${time} all contacts good after ${formatSeconds(event.duration_seconds)}`;
  }
  const channels = (event.channels || []).join(", ") || "contact";
  const duration = event.duration_seconds == null ? "" : ` for ${formatSeconds(event.duration_seconds)}`;
  const reason = (event.reason_codes || []).length > 0 ? ` - ${primaryReason(event.reason_codes)}` : "";
  return `${time} ${channels} dropped${duration}${reason}`;
}

function formatClock(timestampSeconds) {
  if (timestampSeconds == null) {
    return "--:--:--";
  }
  return new Date(timestampSeconds * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function formatDuration(seconds) {
  const value = numberOrNull(seconds);
  if (value == null) {
    return "00:00";
  }
  const total = Math.max(0, Math.floor(value));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${pad(minutes)}:${pad(secs)}`;
  }
  return `${pad(minutes)}:${pad(secs)}`;
}

function formatTimeOfDay(timestampSeconds) {
  return new Date(timestampSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatMinutes(seconds) {
  const value = numberOrNull(seconds);
  if (value == null) {
    return null;
  }
  const minutes = Math.round(value / 60);
  if (minutes < 1) {
    return "under a minute";
  }
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const rest = minutes % 60;
  return rest ? `${Math.floor(minutes / 60)} h ${rest} min` : `${Math.floor(minutes / 60)} h`;
}

// Lost or repeated subpackets per sensor type, summed; null before the decoder counted them.
function subpacketTotal(counts) {
  if (!counts || typeof counts !== "object") {
    return null;
  }
  return Object.values(counts).reduce((sum, count) => sum + (Number(count) || 0), 0);
}

function formatSeconds(seconds) {
  const value = numberOrNull(seconds);
  return value == null ? "--" : `${Math.max(0, value).toFixed(1)}s`;
}

function formatAge(seconds) {
  const value = numberOrNull(seconds);
  if (value == null) {
    return "-";
  }
  if (value < 1) {
    return `${Math.round(value * 1000)} ms ago`;
  }
  return `${value.toFixed(1)} s ago`;
}

function formatRate(value, unit = "Hz") {
  const number = numberOrNull(value);
  return number == null ? `-- ${unit}` : `${number.toFixed(1)} ${unit}`;
}

function formatPercent(value) {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

function formatUnknownTags(tags) {
  const entries = Object.entries(tags);
  if (entries.length === 0) {
    return "0";
  }
  const total = entries.reduce((sum, [, count]) => sum + Number(count || 0), 0);
  return `${total} (${entries.map(([tag, count]) => `${tag}: ${count}`).join(", ")})`;
}

function titleCase(value) {
  const text = String(value || "");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function pad(value) {
  return String(value).padStart(2, "0");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function numberOrNull(value) {
  if (value == null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
