// DOM Elements
const display = document.getElementById('display');
const digitButtons = document.querySelectorAll('[data-digit]');
const opButtons = document.querySelectorAll('[data-op]');
const clearBtn = document.querySelector('[data-clear]');
const equalsBtn = document.querySelector('[data-equals]');

// State variables
let currentInput = '';
let previousOperand = null;
let operator = null;
let shouldResetDisplay = false;

// Helper functions
function updateDisplay() {
  display.textContent = currentInput || '0';
}

function appendDigit(digit) {
  if (shouldResetDisplay) {
    currentInput = '';
    shouldResetDisplay = false;
  }
  // Handle decimal point
  if (digit === '.' && currentInput.includes('.')) return;
  // Handle leading zeros
  if (currentInput === '0' && digit !== '.') {
    currentInput = digit;
  } else {
    currentInput += digit;
  }
  updateDisplay();
}

function chooseOperator(op) {
  // Normalize operator symbols from UI to internal representation
  const opMap = {
    '+': '+',
    '-': '-',
    '×': '*',
    '÷': '/'
  };
  const internalOp = opMap[op] || op;

  if (currentInput === '') return;

  if (previousOperand !== null) {
    compute();
  }

  previousOperand = parseFloat(currentInput);
  operator = internalOp;
  shouldResetDisplay = true;
}

function compute() {
  if (operator === null || currentInput === '') return;
  const prev = previousOperand;
  const current = parseFloat(currentInput);
  let result;
  switch (operator) {
    case '+':
      result = prev + current;
      break;
    case '-':
      result = prev - current;
      break;
    case '*':
      result = prev * current;
      break;
    case '/':
      if (current === 0) {
        result = 'Error';
      } else {
        result = prev / current;
      }
      break;
    default:
      return;
  }
  currentInput = result.toString();
  previousOperand = null;
  operator = null;
  shouldResetDisplay = true;
  updateDisplay();
}

function clearAll() {
  currentInput = '';
  previousOperand = null;
  operator = null;
  shouldResetDisplay = false;
  updateDisplay();
}

// Event listeners (defined after helper functions)

digitButtons.forEach(button => {
  button.addEventListener('click', e => {
    e.preventDefault();
    appendDigit(button.dataset.digit);
  });
});

opButtons.forEach(button => {
  button.addEventListener('click', e => {
    e.preventDefault();
    chooseOperator(button.dataset.op);
  });
});

clearBtn.addEventListener('click', e => {
  e.preventDefault();
  clearAll();
});

equalsBtn.addEventListener('click', e => {
  e.preventDefault();
  compute();
});

// Initialize display
updateDisplay();
