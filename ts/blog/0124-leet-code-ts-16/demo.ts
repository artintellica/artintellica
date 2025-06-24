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
