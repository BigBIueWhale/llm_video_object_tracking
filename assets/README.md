# Fonts in `assets/`

This folder contains the two DejaVu TTF fonts used by the PDF generator.
They were obtained from the official DejaVu releases and copied here intact.

## Sources (official)

* **DejaVu Sans (ZIP of TTFs)** — contains `DejaVuSans.ttf`
  [https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-sans-ttf-2.37.zip/download](https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-sans-ttf-2.37.zip/download)

* **DejaVu full TTF pack (tar.bz2)** — contains `DejaVuSansMono.ttf` (along with other faces)
  [https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-fonts-ttf-2.37.tar.bz2/download](https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-fonts-ttf-2.37.tar.bz2/download)

* **DejaVu / Bitstream Vera license (full text)**
  [https://dejavu-fonts.github.io/License.html](https://dejavu-fonts.github.io/License.html)

## What was extracted

From the two archives above, only the regular Sans and regular Sans Mono were taken.

### Provenance → Destination

| Source archive / path                          | File copied to this repo      |
| ---------------------------------------------- | ----------------------------- |
| `dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf`      | `./assets/DejaVuSans.ttf`     |
| `dejavu-fonts-ttf-2.37/ttf/DejaVuSansMono.ttf` | `./assets/DejaVuSansMono.ttf` |

* No other weights/styles (Serif, Condensed, Bold, etc.) were copied.

## Current contents

```
./assets/
  DejaVuSans.ttf
  DejaVuSansMono.ttf
  LICENSES.md
  README.md  (this file)
```

## License

* The DejaVu family is distributed under the Bitstream Vera–derived license.
* You may redistribute the fonts with notice; do **not** sell the fonts by themselves; modified fonts must be renamed.
* `LICENSES.md` in this folder contains the full license text copied from the official DejaVu License page.

## Copy commands

### Windows (Command Prompt)

```cmd
copy ".\assets\dejavu-sans-ttf-2.37\ttf\DejaVuSans.ttf" ".\assets\DejaVuSans.ttf"
copy ".\assets\dejavu-fonts-ttf-2.37\ttf\DejaVuSansMono.ttf" ".\assets\DejaVuSansMono.ttf"
copy ".\assets\dejavu-fonts-ttf-2.37\LICENSE" ".\assets\LICENSES.md"
```

### macOS / Linux

```bash
cp ./assets/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf ./assets/DejaVuSans.ttf
cp ./assets/dejavu-fonts-ttf-2.37/ttf/DejaVuSansMono.ttf ./assets/DejaVuSansMono.ttf
cp ./assets/dejavu-fonts-ttf-2.37/LICENSE ./assets/LICENSES.md
```
