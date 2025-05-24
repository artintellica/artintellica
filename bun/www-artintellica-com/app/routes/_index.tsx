import { useState } from "react";
import { twMerge } from "tailwind-merge";
import { $aicon } from "~app/util/aicons";

export default function IndexPage() {
  const [searchInput, setSearchInput] = useState("");

  function performSearch() {
    console.log("Performing search for:", searchInput);
  }

  return (
    <div>
      <div className="my-4 mx-auto block aspect-square w-[120px]">
        <img
          draggable={false}
          src={$aicon("/images/orange-cat-robot-300.webp")}
          alt=""
          className="block"
        />
      </div>
      <h1 className="mt-4 text-center font-bold text-2xl text-black dark:text-white">
        Artintellica
      </h1>
      <h2 className="my-4 text-center text-black dark:text-white">
        Open-source math and code for AI.
      </h2>
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div className="flex gap-x-1">
          <div className="relative max-w-[600px] flex-grow">
            <label htmlFor="primary-search">
              <img
                src={$aicon("/images/magnifying-glass-64.webp")}
                alt="Magnifying Glass"
                draggable="false"
                className="absolute top-[8px] left-[8px] h-[20px] w-[20px]"
              />
            </label>
            <input
              autoFocus
              id="primary-search"
              name="q"
              type="text"
              placeholder="Search Artintellica"
              onChange={(e) => {
                setSearchInput(e.target.value);
              }}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  performSearch();
                }
              }}
              value={searchInput}
              className={twMerge(
                "w-full grow overflow-hidden rounded-lg border-none bg-gray-100 p-2 pl-[36px] text-black text-sm shadow-sm ring-1 ring-gray-300 ring-inset focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset dark:bg-gray-900 dark:text-white dark:ring-gray-700",
              )}
            />
          </div>
          <div className="flex-shrink-0">
            <button
              type="button"
              onClick={performSearch}
              className="h-full cursor-pointer rounded-lg bg-gray-200 p-2 font-bold text-gray-600 text-sm shadow-sm hover:bg-blue-500 hover:text-white dark:bg-gray-800 dark:text-gray-400 dark:hover:border-white dark:hover:text-white"
            >
              Search
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
