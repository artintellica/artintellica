// pairs: ()[]{}
// no other allowed characters

function isValid(str: string) {
  const stack: string[] = [];

  for (const char of str) {
    if (char === "(") {
      stack.push("(");
    } else if (char === "[") {
      stack.push("[");
    } else if (char === "{") {
      stack.push("{");
    } else if (char === ")") {
      if (stack[stack.length - 1] !== "(") {
        return false;
      }
      stack.pop();
    } else if (char === "]") {
      if (stack[stack.length - 1] !== "[") {
        return false;
      }
      stack.pop();
    } else if (char === "}") {
      if (stack[stack.length - 1] !== "{") {
        return false;
      }
      stack.pop();
    } else {
      return false;
    }
  }
  if (stack.length !== 0) {
    return false;
  }
  return true;
}

// Test cases to demonstrate the solution
console.log(isValid("()")); // true
console.log(isValid("()[]{}")); // true
console.log(isValid("(]")); // false
console.log(isValid("([)]")); // false
console.log(isValid("{[]}")); // true
