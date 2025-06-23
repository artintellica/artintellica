function isValid(s: string): boolean {
  // Define a mapping of closing brackets to opening brackets using a Map
  const bracketsMap = new Map<string, string>([
    [")", "("],
    ["}", "{"],
    ["]", "["],
  ]);

  // Stack to store opening brackets
  const stack: string[] = [];

  // Iterate through each character in the input string
  for (const char of s) {
    // If the character is a closing bracket (exists as a key in the map)
    if (bracketsMap.has(char)) {
      // Check if stack is empty (no opening bracket to match) or
      // if the top of the stack doesn't match the expected opening bracket
      if (
        stack.length === 0 ||
        stack[stack.length - 1] !== bracketsMap.get(char)
      ) {
        return false;
      }
      // If it matches, pop the top opening bracket from the stack
      stack.pop();
    } else {
      // If it's an opening bracket, push it onto the stack
      stack.push(char);
    }
  }

  // Return true only if all brackets are matched (stack is empty)
  return stack.length === 0;
}

// Test cases to demonstrate the solution
console.log(isValid("()")); // true
console.log(isValid("()[]{}")); // true
console.log(isValid("(]")); // false
console.log(isValid("([)]")); // false
console.log(isValid("{[]}")); // true
