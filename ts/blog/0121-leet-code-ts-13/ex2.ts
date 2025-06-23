type Pairs = Map<string, string>;

function isValid(str: string, pairs: Pairs) {
  const stack: string[] = [];
  for (const char of str) {
    if (pairs.has(char)) {
      stack.push(char);
    }
    // otherwise, we must be closing the top item, or else...
    else {
      const top = stack[stack.length - 1];
      const right = pairs.get(top);
      if (char !== right) {
        return false;
      }
      stack.pop();
    }
  }
  if (stack.length !== 0) {
    return false;
  }
  return true;
}

const pairs: Pairs = new Map();
pairs.set("(", ")");
pairs.set("[", "]");
pairs.set("{", "}");
pairs.set("<", ">");

// Test cases to demonstrate the solution
console.log(isValid("()", pairs)); // true
console.log(isValid("()[]{}", pairs)); // true
console.log(isValid("(]", pairs)); // false
console.log(isValid("([)]", pairs)); // false
console.log(isValid("{[]}", pairs)); // true
