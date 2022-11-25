# Server

- Connect to the server
  ```bash
  ssh <your_itsc>@ugcpu4.cse.ust.hk
  ```
- For the first time you log in
  ```bash
  cd /localdata
  mkdir <your_own_folder>
  ```
- Store your files here
  ```bash
  cd /localdata
  ```
- Do this **every time** you log in the server
  ```bash
  setenv PYTHONUSERBASE /localdata/<your_own_folder>
  ```

P.S. In case the storage (of `/localdata`) is running out

- Check disk capacity
  ```bash
  df -h
  ```
- Switch to `/data`: replace all the `/localdata` above by `/data`

Copy (large) files using `scp`:

- Basic syntax
  ```bash
  scp <source> <destination>
  ```
- From your local machine to a remote server
  ```bash
  scp /path/to/source <username>@<server>:/path/to/destination
  ```
- From a remote to your local machine
  ```bash
  scp <username>@<server>:/path/to/destination /path/to/source
  ```

Edit file using:

- `nano <file>`: see the prompt at the bottom
  - Navigate: Ctrl-n (next line), Ctrl-p (previous line), Ctrl-b (move backward),
    Ctrl-f (move forward), Ctrl-a (jump to the beginning of the line), Ctrl-e
    (jump to the end)
  - Save and exit: Ctrl-o (write out), press Enter to confirm the file name,
    Ctrl-x (exit)
- `emacs <file>`:
  - exit by Ctrl-x Ctrl-c
- `vi <file>` or `vim <file>`

# Python Packages

Need to use `--user` flag with `pip install`

- upgrade pip
  ```bash
  python-csd-3.10 -m pip install --user --upgrade pip
  ```
- install from requirement.txt
  ```bash
  python-csd-3.10 -m pip install --user -r requirements.txt
  ```
- output installed package
  ```bash
  python-csd-3.10 -m pip list --format=freeze > requirements.txt
  ```
- add a directory to PATH
  ```bash
  bash -c "export PATH=<directory>:$PATH"
  ```
