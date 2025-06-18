import blogPosts from "~app/blog.json";
import { Footer } from "~app/components/footer.js";
import { $aicon } from "~app/util/aicons";
import type { Route } from "./+types/_index.js";
import { Link, href } from "react-router";

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
    .slice(0, 20); // Limit to the most recent 20 posts
  return { blogPosts: newBlogPosts };
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
  const { blogPosts } = loaderData;
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
      <Footer />
    </div>
  );
}
