# ml-lab-improved

## Setup: Open project in VS Code and create Python environment

**Step 3: Open the Project in VS Code**

- **Open VS Code:** Launch Visual Studio Code.
- **Open Folder:** Go to `File > Open Folder...` (or press `Ctrl+K` `Ctrl+O` on Windows).
- **Select project folder:** Navigate to and select the `ml-lab-improved` folder created when you cloned the repo.

**Step 4: Set up the Python Environment**

- **Open the integrated terminal:** `Terminal > New Terminal` (or press ``Ctrl+`` `` ).
- **Create a virtual environment (recommended):**

	- Windows PowerShell (create):

		```powershell
		python -m venv .venv
		```

	- Activate the venv on Windows PowerShell:

		```powershell
		.\.venv\Scripts\Activate
		```

- **Install dependencies:** If this project includes a `requirements.txt` file, install with:

	```powershell
	pip install -r requirements.txt
	```

- **Select the interpreter in VS Code:** Click the Python version in the bottom-left and choose the Python executable inside `./.venv` (Windows path: `.\.venv\Scripts\python.exe`).

- **Run notebooks / scripts:** You can start Jupyter notebooks with:

	```powershell
	pip install notebook
	jupyter notebook
	```

> Note: Update `requirements.txt` with the exact libraries your project needs. A minimal example `requirements.txt` is included in this repo to get started.
