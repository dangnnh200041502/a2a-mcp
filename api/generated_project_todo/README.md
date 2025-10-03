# SimpleTodoApp

## Project Overview
SimpleTodoApp is a lightweight, browser‑based todo list application that lets users manage their tasks efficiently. It runs entirely on the client side, storing data in the browser's `localStorage`, which means no server setup or database is required. The app provides a clean UI, keyboard shortcuts, and filtering options to keep users productive.

---

## Features
- **Add tasks** – Quickly create new todos with a title and optional due date.
- **Edit tasks** – Inline editing of task titles and due dates.
- **Delete tasks** – Remove tasks you no longer need.
- **Mark as complete** – Toggle a task's completion status.
- **Filter view** – Switch between All, Active, and Completed tasks.
- **Keyboard shortcuts** – 
  - `Enter` – Add a new task when the input field is focused.
  - `Esc` – Cancel editing.
  - `Ctrl + A` – Focus the new‑task input.
- **Persistence** – All tasks are saved to `localStorage` and restored on page reload.
- **Responsive design** – Works on desktop, tablet, and mobile devices.
- **No external dependencies** – Pure HTML, CSS, and vanilla JavaScript.

---

## Tech Stack
| Layer | Technology |
|-------|------------|
| Markup | **HTML5** – `index.html` |
| Styling | **CSS3** – `styles.css` |
| Logic | **JavaScript (ES6+)** – `app.js` |
| Persistence | **Web Storage API** (`localStorage`) |

---

## Installation & Running
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/simpletodoapp.git
   cd simpletodoapp
   ```
2. **Open the app** – No build step is required. Simply open `index.html` in a modern browser:
   ```bash
   # On macOS/Linux
   open index.html

   # On Windows
   start index.html
   ```
   Or double‑click the file from your file explorer.
3. The application will load, and you can start adding tasks immediately.

---

## Usage
### Adding a Task
1. Click the input field at the top or press **Ctrl + A** to focus it.
2. Type the task description.
3. Press **Enter** to add the task to the list.

### Editing a Task
- Double‑click the task text or click the pencil icon (if present).
- Modify the text and press **Enter** to save or **Esc** to cancel.

### Deleting a Task
- Click the trash‑can icon next to the task.
- The task is removed instantly and the change is persisted.

### Completing a Task
- Click the checkbox on the left of a task.
- Completed tasks are visually crossed out and moved to the *Completed* filter view.

### Filtering Tasks
- Use the navigation bar at the bottom to switch between:
  - **All** – Shows every task.
  - **Active** – Shows only incomplete tasks.
  - **Completed** – Shows only completed tasks.

### Keyboard Shortcuts Summary
| Shortcut | Action |
|----------|--------|
| `Ctrl + A` | Focus the new‑task input field |
| `Enter` (input) | Add a new task |
| `Enter` (editing) | Save edited task |
| `Esc` (editing) | Cancel editing |
| `Space` (checkbox) | Toggle completion |

---

## Code Structure
```
/simpletodoapp
│   README.md          # ← This documentation
│   index.html         # Main HTML page
│   styles.css         # Application styling
│   app.js             # Core JavaScript logic
│
└───assets/            # Optional images, icons, etc.
```

### `index.html`
- Sets up the DOM structure: header, input field, task list container, and footer navigation.
- Links `styles.css` and `app.js`.

### `styles.css`
- Provides a clean, modern UI using Flexbox and CSS variables.
- Contains media queries for responsive behavior on mobile devices.

### `app.js`
- **Key Functions / Classes**
  - `loadTasks()` – Retrieves the task array from `localStorage` (or returns an empty array).
  - `saveTasks(tasks)` – Serialises the task array and stores it in `localStorage`.
  - `renderTasks(filter = 'all')` – Renders tasks based on the selected filter.
  - `addTask(title)` – Creates a new task object, pushes it to the task list, and persists it.
  - `editTask(id, newTitle)` – Updates a task's title.
  - `toggleComplete(id)` – Flips the `completed` flag.
  - `deleteTask(id)` – Removes a task from the array.
  - `setupEventListeners()` – Binds UI events (clicks, keypresses, filter changes).
- Uses the **module pattern** to keep the global namespace clean.

---

## Persistence Details
Tasks are stored as a JSON string under the key `simpleTodoTasks` in the browser's `localStorage`.
```js
// Example of the stored value
localStorage.setItem('simpleTodoTasks', JSON.stringify(tasksArray));
```
When the app loads, `app.js` calls `loadTasks()` to parse this JSON back into an array of task objects.

### Task Object Structure
```json
{
  "id": "a1b2c3d4",   // Unique identifier (generated via Date.now() or crypto.randomUUID())
  "title": "Buy groceries",
  "completed": false,
  "createdAt": 1727184000000, // Unix timestamp
  "dueDate": "2025-10-01"    // Optional ISO date string, may be null
}
```
All modifications (add, edit, toggle, delete) immediately call `saveTasks()` to keep `localStorage` in sync.

---

## Responsiveness
- **Desktop (≥ 768px)** – Two‑column layout with the task list centered.
- **Tablet (480 – 767px)** – Full‑width list, larger tap targets.
- **Mobile (< 480px)** – Stacked layout, input field expands to full width, navigation icons become larger for touch.
- Media queries are defined in `styles.css` to adjust font sizes, spacing, and element widths.

---

## Contribution Guidelines
1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/simpletodoapp.git
   cd simpletodoapp
   ```
3. Create a new branch for your feature or bug‑fix:
   ```bash
   git checkout -b feature/awesome-feature
   ```
4. Make your changes, ensuring the code follows the existing style (ES6, semicolons, meaningful variable names).
5. **Test** the app by opening `index.html` in a browser.
6. Commit and push your branch:
   ```bash
   git add .
   git commit -m "Add awesome feature"
   git push origin feature/awesome-feature
   ```
7. Open a **Pull Request** on the original repository and describe the changes.

### Code Style
- Use **2‑space indentation**.
- Prefer `const`/`let` over `var`.
- Keep functions short and focused.
- Add comments for any non‑obvious logic.

---

## License
SimpleTodoApp is licensed under the **MIT License**. See the `LICENSE` file for full details.

---

*Happy task‑tracking!*