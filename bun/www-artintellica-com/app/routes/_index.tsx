import { Footer } from "~app/components/footer.js";
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
        Open-source educational resources for AI.
      </h2>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <Footer />
    </div>
  );
}
