
## Example Commands

### Using URL Input

```bash
Microlensify urls_file.txt yes 2
```
- Flux is already normalized, so the model uses fixed statistics from the training set.

- Uses 2 CPU cores.

### Using Local Files

```bash
Microlensify OGLE_filelist.txt yes 16
```
- Computes flux statistics from flux.

- Uses 16 CPU cores.
