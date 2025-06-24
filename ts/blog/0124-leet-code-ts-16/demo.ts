function lengthOfLongestSubstring(s: string): number {
  // Initialize a map to store the last index of each character
  const charIndexMap: Map<string, number> = new Map();

  // Initialize variables for the left pointer, max length, and loop through string with right pointer
  let left = 0;
  let maxLength = 0;

  // Iterate over the string using the right pointer
  for (let right = 0; right < s.length; right++) {
    const currentChar: string = s[right] as string;

    // Check if the current character is already in the window
    if (charIndexMap.has(currentChar)) {
      // Move the left pointer to the position just after the last occurrence of currentChar
      // Use Math.max to ensure we don't move left pointer backward
      left = Math.max(left, charIndexMap.get(currentChar)! + 1);
    }

    // Update the last seen index of the current character
    charIndexMap.set(currentChar, right);

    // Update the maximum length if the current window is larger
    maxLength = Math.max(maxLength, right - left + 1);
  }

  return maxLength;
}

function runTests(): void {
  const testCases: { input: string; expected: number }[] = [
    { input: "abcabcbb", expected: 3 }, // "abc"
    { input: "bbbbb", expected: 1 }, // "b"
    { input: "pwwkew", expected: 3 }, // "wke"
    { input: "", expected: 0 }, // empty string
    { input: "dvdf", expected: 3 }, // "vdf"
  ];

  testCases.forEach((test, index) => {
    const result: number = lengthOfLongestSubstring(test.input);
    console.log(
      `Test ${index + 1}: Input="${test.input}", Expected=${test.expected}, Got=${result}, ${
        result === test.expected ? "PASS" : "FAIL"
      }`,
    );
  });
}

// Run the tests
runTests();
