import blogPosts from "~app/blog.json";
import { Footer } from "~app/components/footer.js";
import { $aicon } from "~app/util/aicons";
import type { Route } from "./+types/_index.js";
import { Link, href } from "react-router";
import seriesPosts from "~app/series.json";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
  const newBlogPosts = blogPosts
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
    .reverse()
    .slice(0, 10); // Limit to the most recent 10 posts
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
    .reverse()
    .slice(0, 10); // Limit to the most recent 10 series posts
  return { blogPosts: newBlogPosts, seriesPosts: newSeriesPosts };
};

export const meta: Route.MetaFunction = () => {
  return [
    { title: "Artintellica: Learn machine learning with AI." },
    {
      name: "description",
      content: "Learn machine learning with AI.",
    },
  ];
};

export default function IndexPage({ loaderData }: Route.ComponentProps) {
  const { blogPosts, seriesPosts } = loaderData;
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
      <div className="my-4 hidden dark:block">
        <img
          draggable={false}
          src={"/images/artintellica-text-white.png"}
          alt="Artintellica"
          className="mx-auto block w-[250px]"
        />
      </div>
      <div className="my-4 block dark:hidden">
        <img
          draggable={false}
          src={"/images/artintellica-text-black.png"}
          alt="Artintellica"
          className="mx-auto block w-[250px]"
        />
      </div>
      <h2 className="my-4 text-center text-black dark:text-white">
        Learn machine learning with AI.
      </h2>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div>
          <h1 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Recent Series Posts
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
          to={href("/series")}
        >
          Browse All Series Posts
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div>
          <h1 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Recent Blog Posts
          </h1>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {blogPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/blog/:filename", { filename: post.filename })}
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
      <div className="text-center text-black dark:text-white">
        <Link
          className="border-b border-b-blue font-bold text-lg hover:border-b-black dark:hover:border-b-white"
          to={href("/blog")}
        >
          Browse All Blog Posts
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <Footer />
    </div>
  );
}
