// script.js - SimpleTodoApp core logic

// Ensure the script runs after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // =============================
  // 1. DOM References & Constants
  // =============================
  const newTaskInput = document.getElementById('new-task-input');
  const addTaskBtn = document.getElementById('add-task-btn');
  const taskList = document.getElementById('task-list');
  const filterAllBtn = document.getElementById('filter-all');
  const filterActiveBtn = document.getElementById('filter-active');
  const filterCompletedBtn = document.getElementById('filter-completed');

  const FILTER = {
    ALL: 'all',
    ACTIVE: 'active',
    COMPLETED: 'completed',
  };

  let currentFilter = FILTER.ALL;

  // ==================
  // 2. Task Model
  // ==================
  class TodoTask {
    constructor(id, text, completed = false) {
      this.id = id;
      this.text = text;
      this.completed = completed;
    }

    toggleCompleted() {
      this.completed = !this.completed;
    }

    setText(newText) {
      this.text = newText;
    }
  }

  // ==============================
  // 3. LocalStorage Persistence
  // ==============================
  let tasks = [];

  function loadTasks() {
    const stored = localStorage.getItem('simpleTodoTasks');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        // Recreate TodoTask instances (optional, but keeps methods available)
        tasks = parsed.map(obj => new TodoTask(obj.id, obj.text, obj.completed));
      } catch (e) {
        console.error('Failed to parse stored tasks:', e);
        tasks = [];
      }
    }
  }

  function saveTasks() {
    // Store plain objects (JSON.stringify will ignore class methods automatically)
    const plain = tasks.map(t => ({ id: t.id, text: t.text, completed: t.completed }));
    localStorage.setItem('simpleTodoTasks', JSON.stringify(plain));
  }

  // ==================
  // 4. Render Function
  // ==================
  function renderTasks() {
    // Clear existing list
    taskList.innerHTML = '';

    const filtered = tasks.filter(task => {
      if (currentFilter === FILTER.ALL) return true;
      if (currentFilter === FILTER.ACTIVE) return !task.completed;
      if (currentFilter === FILTER.COMPLETED) return task.completed;
      return true;
    });

    filtered.forEach(task => {
      const li = document.createElement('li');
      li.className = 'task-item';
      if (task.completed) li.classList.add('completed');
      li.dataset.id = task.id;

      // Checkbox for completion
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'toggle-complete';
      checkbox.checked = task.completed;
      // Toggle handler
      checkbox.addEventListener('change', () => {
        toggleTaskCompletion(task.id);
      });

      // Text span
      const span = document.createElement('span');
      span.className = 'task-text';
      span.textContent = task.text;

      // Edit button
      const editBtn = document.createElement('button');
      editBtn.className = 'edit-btn';
      editBtn.textContent = 'Edit';
      editBtn.addEventListener('click', () => {
        startEditMode(li, task);
      });

      // Delete button
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'delete-btn';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => {
        deleteTask(task.id);
      });

      // Assemble li
      li.appendChild(checkbox);
      li.appendChild(span);
      li.appendChild(editBtn);
      li.appendChild(deleteBtn);

      taskList.appendChild(li);
    });

    // Update filter button active state
    updateFilterButtons();
  }

  // Helper to update active filter button styling
  function updateFilterButtons() {
    const btnMap = {
      [FILTER.ALL]: filterAllBtn,
      [FILTER.ACTIVE]: filterActiveBtn,
      [FILTER.COMPLETED]: filterCompletedBtn,
    };
    Object.values(btnMap).forEach(btn => btn.classList.remove('active'));
    const activeBtn = btnMap[currentFilter];
    if (activeBtn) activeBtn.classList.add('active');
  }

  // ==================
  // 5. Add Task
  // ==================
  function addTask(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    const id = Date.now().toString(); // simple unique id
    const newTask = new TodoTask(id, trimmed);
    tasks.push(newTask);
    saveTasks();
    renderTasks();
    newTaskInput.value = '';
    newTaskInput.focus();
  }

  // Bind add button click
  addTaskBtn.addEventListener('click', () => {
    addTask(newTaskInput.value);
  });

  // Bind Enter key on input
  newTaskInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTask(newTaskInput.value);
    }
  });

  // ==================
  // 6. Edit Task
  // ==================
  function editTask(id, newText) {
    const task = tasks.find(t => t.id === id);
    if (!task) return;
    const trimmed = newText.trim();
    if (trimmed) task.setText(trimmed);
    saveTasks();
    renderTasks();
  }

  // Enter edit mode – replace span with input
  function startEditMode(liElement, task) {
    const span = liElement.querySelector('.task-text');
    if (!span) return;

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'edit-input';
    input.value = task.text;
    input.style.flex = '1';
    // Replace span with input
    liElement.replaceChild(input, span);
    input.focus();
    input.select();

    // Commit edit on Enter or blur
    const commit = () => {
      editTask(task.id, input.value);
    };
    const cancel = () => {
      // Restore original span without changes
      renderTasks();
    };

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        commit();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        cancel();
      }
    });
    input.addEventListener('blur', commit);
  }

  // ==================
  // 7. Delete Task
  // ==================
  function deleteTask(id) {
    tasks = tasks.filter(t => t.id !== id);
    saveTasks();
    renderTasks();
  }

  // ==============================
  // 8. Toggle Completion
  // ==============================
  function toggleTaskCompletion(id) {
    const task = tasks.find(t => t.id === id);
    if (!task) return;
    task.toggleCompleted();
    saveTasks();
    renderTasks();
  }

  // ==================
  // 9. Filtering
  // ==================
  function setFilter(filter) {
    if (!Object.values(FILTER).includes(filter)) return;
    currentFilter = filter;
    renderTasks();
  }

  // Bind filter buttons
  filterAllBtn.addEventListener('click', () => setFilter(FILTER.ALL));
  filterActiveBtn.addEventListener('click', () => setFilter(FILTER.ACTIVE));
  filterCompletedBtn.addEventListener('click', () => setFilter(FILTER.COMPLETED));

  // ==============================
  // 10. Keyboard Shortcuts
  // ==============================
  document.addEventListener('keydown', e => {
    // Ignore when focus is on an input/textarea to avoid interfering with typing
    const activeEl = document.activeElement;
    const isInput = activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA');

    if (!isInput && e.key === '/') {
      e.preventDefault();
      newTaskInput.focus();
    }
  });

  // Initial load & render
  loadTasks();
  renderTasks();
});
