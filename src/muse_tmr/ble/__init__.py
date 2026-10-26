"""The forked amused-py BLE layer: connecting to the Muse S Athena, decoding its packets, raw files.

The modules came from the repository root in #232 with only their imports of each other
changed; docs/dependency_strategy.md says how upstream amused-py patches map onto them.
Import the modules themselves, for example ``muse_tmr.ble.muse_stream_client``. This
package imports none of them, so loading the raw file reader does not load bleak.
"""
