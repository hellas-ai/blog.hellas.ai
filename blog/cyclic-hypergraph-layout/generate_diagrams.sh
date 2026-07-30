#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json '([x y . x y x y] {xor and})' > 1-bit-adder.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition nor > nor.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition latch > latch.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --mode frobenius --definition latch > latch-frobenius.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition cycle-not > cycle-not.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition cycle-not-cupcap > cycle-not-cupcap.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels '[. x]' > frobenius-unit.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels '[x . x x]' > frobenius-split.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels '[x x . x]' > frobenius-merge.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels '[x .]' > frobenius-counit.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition compose-f-g > compose-f-g.svg
typed-pictures circuit/circuit.hex --padding 0.25 --no-wire-labels --generators circuit/circuit-generators.json --definition compose-f-g-frobenius > compose-f-g-frobenius.svg
