import { $aicon } from "~app/util/aicons";

export default function IndexPage() {
  return (
    <div>
      <div className="mx-auto my-4 block aspect-square w-[120px]">
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
    </div>
  );
}
