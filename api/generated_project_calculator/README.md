# Simple Calculator

A lightweight, responsive calculator built with vanilla HTML, CSS, and JavaScript. It features a clean UI, easy-to-use buttons, and works on both desktop and mobile browsers without any build steps.

## Features
- **Basic arithmetic**: addition, subtraction, multiplication, and division.
- **Clear button**: resets the current expression.
- **Equals button**: evaluates the entered expression.
- **Responsive layout**: adapts to various screen sizes.
- **Accessible**: ARIA labels and live region for screen readers.
- **No external dependencies**: pure JavaScript and CSS.

## Demo
You can view a live demo here: [Simple Calculator Demo](https://example.com) *(Replace with actual link if available)*

![Calculator Screenshot](screenshot.png)

## Setup
No build steps are required. Just open `index.html` in a modern web browser.

```bash
# From the project root
open index.html   # macOS
# or
start index.html  # Windows
# or
xdg-open index.html  # Linux
```

## Usage
1. **Digits** – Click or tap the numbers 0‑9 to build your expression.
2. **Operations** – Use `+`, `-`, `×`, and `÷` to add operators.
3. **Clear** – Press the `C` button to reset the calculator.
4. **Equals** – Press `=` to evaluate the current expression. The result is displayed in the panel.

The calculator updates the display in real time as you input values.

## Development
The project is intentionally minimal. If you wish to extend it:

- **Add more operations** – Update `app.js` to handle new operators and add corresponding buttons in `index.html`.
- **Improve styling** – Edit `styles.css` or add new variables.
- **Add unit tests** – Create a `tests/` folder and write tests using Jest or another framework.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
