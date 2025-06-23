function isValidExtended(s: string, pairs: [string, string][]): boolean {
  // Create a map from closing to opening characters based on provided pairs
  const charMap = new Map<string, string>();
  for (const [open, close] of pairs) {
    charMap.set(close, open);
  }

  const stack: string[] = [];
  const openingChars = new Set(pairs.map((pair) => pair[0]));

  for (const char of s) {
    if (charMap.has(char)) {
      if (stack.length === 0 || stack[stack.length - 1] !== charMap.get(char)) {
        return false;
      }
      stack.pop();
    } else if (openingChars.has(char)) {
      stack.push(char);
    }
  }

  return stack.length === 0;
}

// Test cases for extended validation
const bracketPairs: [string, string][] = [
  ["(", ")"],
  ["{", "}"],
  ["[", "]"],
  ["<", ">"],
];
console.log(isValidExtended("()<>", bracketPairs)); // true
console.log(isValidExtended("(<>)", bracketPairs)); // true
console.log(isValidExtended("(<)", bracketPairs)); // false
console.log(isValidExtended("abc()def", bracketPairs)); // true (ignores non-paired chars)
