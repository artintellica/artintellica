function wordBreak(str: string, wordDict: string[]): boolean {
  //strategy:
  //first, put wordDict into a set for fast look ups
  //second, loop through substrings of str starting from 0 to N
  //fill dp array with whether string can be broken

  const wordSet = new Set<string>(wordDict);
  const dp: boolean[] = new Array(str.length + 1).fill(false);

  dp[0] = true; // empty string is always valid

  // iterate through each position idx in the string
  for (let idx = 1; idx <= str.length; idx++) {
    // check every possible substring ending at idx by looking at previous positions jdx
    for (let jdx = 0; jdx < idx; jdx++) {
      // if s[0...jdx-1] is valid and s[j...i-1] is in the dictionary
      if (dp[jdx] && wordSet.has(str.substring(jdx, idx))) {
        dp[idx] = true;
        break; // no need to check further once we find a valid split
      }
    }
  }

  return dp[str.length] as boolean;
}

// Test cases to demonstrate the solution
function runTests(): void {
  console.log("Test 1:", wordBreak("leetcode", ["leet", "code"])); // Expected: true
  console.log("Test 2:", wordBreak("applepenapple", ["apple", "pen"])); // Expected: true
  console.log(
    "Test 3:",
    wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]),
  ); // Expected: false
}

// Run the tests
runTests();
