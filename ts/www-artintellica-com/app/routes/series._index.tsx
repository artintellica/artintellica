import seriesPosts from "~app/series.json";
import { Link, href } from "react-router";
import type { Route } from "./+types/series._index.js";
import { $aicon } from "~app/util/aicons.js";
import { Footer } from "~app/components/footer.js";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
  const newSeriesPosts = seriesPosts
    .map((post) => {
      return {
        title: post.title,
        date: post.date,
        author: post.author,
        filename: post.filename,
        icon: post.icon,
        content: "",
      };
    })
    .sort((a, b) => a.filename.localeCompare(b.filename))
    .reverse();
  return { seriesPosts: newSeriesPosts };
};

export const meta: Route.MetaFunction = () => {
  return [
    { title: "Series | Artintellica" },
    { name: "description", content: "Open-source AI resources." },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/json",
      title: "JSON Feed",
      href: "/series/feed.json",
    },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/atom+xml",
      title: "Atom Feed",
      href: "/series/feed.atom.xml",
    },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/rss+xml",
      title: "RSS Feed",
      href: "/series/feed.rss.xml",
    },
  ];
};

export default function SeriesIndex({ loaderData }: Route.ComponentProps) {
  const { seriesPosts } = loaderData;
  return (
    <div>
      <div className="mx-auto my-4 block aspect-square w-[120px]">
        <Link to={href("/")}>
          <img
            draggable={false}
            src={$aicon("/images/orange-cat-robot-300.webp")}
            alt=""
            className="block"
          />
        </Link>
      </div>
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div>
          <h1 className="my-4 text-center font-bold text-2xl text-black dark:text-white">
            Series
          </h1>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {seriesPosts.map((post) => (
                <Link
                  key={post.filename}
                  to={`/series/${post.filename}`}
                  className="my-2 flex space-x-2 rounded-md bg-white/50 p-2 text-black outline-1 outline-black/50 hover:bg-white hover:outline-4 hover:outline-blue dark:bg-black/50 dark:text-white dark:outline-white/50 dark:hover:bg-black"
                >
                  <div className="h-full flex-shrink-0">
                    <img
                      src={`/images/${post.icon}-300.webp`}
                      alt={post.title}
                      className="h-[80px] w-[80px]"
                    />
                  </div>
                  <div className="h-full">
                    <h2 className="my-auto block font-semibold text-lg">
                      {post.title}
                    </h2>
                    <p className="text-black/70 text-sm dark:text-white/70">
                      {post.date} &middot; {post.author}
                    </p>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="text-center text-black dark:text-white">
        <Link
          className="border-b border-b-blue font-bold text-lg hover:border-b-black dark:hover:border-b-white"
          to={href("/")}
        >
          Back to Home
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <Footer />
    </div>
  );
}
