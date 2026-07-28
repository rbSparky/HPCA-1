# FlowAdvantage native bridge patch manifest

Base Morpher commit: `9a9dce7aea521f1d5ef33686f57ca84864edb3c9`.

Built image: `quotientflow-morpher-v5b-native@sha256:6a488771d9cf0aa47dbf0ccd9722948d82df18b6c9e33d5a71ee949536a8c03e`.

Exact source diff: `flowadvantage_bridge.patch`

Patch SHA-256: `2643a94fcbcb1fbe76b318b263a12c74abf9cb5d035fca231a7e0139324d255d`.

The patch adds dump/load flags, actual expanded MRRG export, canonical native-ID mapping export, fresh-MRRG external import, semantic hash guards, directed route validation, and imported binary generation. It does not change PathFinder, SA, LISA, architecture semantics, or the native mapping objective.
