# Third-party notices

## SkillCorner Open Data

This repository includes material derived from SkillCorner Open Data
(https://github.com/SkillCorner/opendata):

- `explorer/data.js` and `docs/index.html` embed 20 tracking frames (player and ball positions,
  velocities computed from them, player names, shirt numbers and positions) from the 2024/25 A-League
  matches released by SkillCorner, rotated and rounded for display.
- `results/` and `figures/` contain statistics and images computed from that data.

The raw data is not redistributed here; `fetch_data.sh` downloads it from SkillCorner's repository.
SkillCorner's licence, reproduced verbatim from their repository:

```
The MIT License

Copyright (c) 2020 SkillCorner. https://skillcorner.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

SkillCorner asks users of the data to credit SkillCorner; this project does so in the README, on every figure and in the sandbox.

## Barlow and Barlow Condensed fonts

`explorer/fonts/*.woff2` (embedded in `docs/index.html`): Copyright 2017 The Barlow Project Authors
(https://github.com/jpt/barlow), licensed under the SIL Open Font License, Version 1.1. The full licence
text is in `explorer/fonts/OFL.txt`. Files obtained from the Fontsource distribution (`@fontsource/barlow`
and `@fontsource/barlow-condensed`, version 5.1.0).
