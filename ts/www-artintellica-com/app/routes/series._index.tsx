import seriesPosts from "~app/series.json";
import { Link, href } from "react-router";
import type { Route } from "./+types/series.js";
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
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/series/:filename", { filename: post.filename })}
                    className="border-b border-b-blue font-semibold text-lg leading-3 hover:border-b-black dark:hover:border-b-white"
                  >
                    {post.title}
                  </Link>
                  <div className="text-black/60 text-sm dark:text-white/60">
                    {post.date} &middot; {post.author}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="text-center text-black text-sm dark:text-white">
        <h2 className="font-bold text-lg">Series Feeds</h2>
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/series/feed.json"}
        >
          JSON Feed
        </a>
        &nbsp;&middot;&nbsp;
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/series/feed.atom.xml"}
        >
          Atom Feed
        </a>
        &nbsp;&middot;&nbsp;
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/series/feed.rss.xml"}
        >
          RSS Feed
        </a>
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
