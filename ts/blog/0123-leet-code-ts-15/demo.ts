function maxProfit(prices: number[]): number {
  // If the array is empty or has only one element, no profit is possible
  if (prices.length < 2) {
    return 0;
  }

  // Initialize variables:
  // minPrice is the lowest price seen so far (best time to buy)
  // maxProfit is the maximum profit we can make
  let minPrice: number = prices[0] as number;
  let maxProfit = 0;

  // Iterate through the array starting from the second element
  for (let i = 1; i < prices.length; i++) {
    const currentPrice = prices[i] as number;

    // Update minPrice if the current price is lower
    if (currentPrice < minPrice) {
      minPrice = currentPrice;
    }
    // Update maxProfit if selling at current price gives a better profit
    else {
      const potentialProfit = currentPrice - minPrice;
      maxProfit = Math.max(maxProfit, potentialProfit);
    }
  }

  return maxProfit;
}
