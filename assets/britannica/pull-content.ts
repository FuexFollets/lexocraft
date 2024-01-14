import HTMLParser, { HTMLElement } from "node-html-parser";
import { convert } from "html-to-text";

const britannicaURL: string = "https://www.britannica.com/";
const britannicaPureDocumentURL: string = "https://www.britannica.com/print/article/";
const britannicaNewArticlesURL: string = "https://www.britannica.com/new-articles";
const britannicaSitemapURL: string = "https://www.britannica.com/sitemap/";
const britannicaSitemapPaths: string[] = [
    "0-9",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
];

async function allSitemapSubpaths(sitemapPath: string): Promise<string[]> {
    const sitemapResponse = await fetch(britannicaURL + sitemapPath);
    const sitemapHTML = HTMLParser.parse(await sitemapResponse.text());

    return sitemapHTML.querySelectorAll("ul.list-unstyled.md-az-browse-content")[0]
        .querySelectorAll("a")
        .map((element) => element.getAttribute("href") as string);
}

async function allArticlePathsFromSitemapSubpath(subpath: string): Promise<string[]> {
    const sitemapResponse = await fetch(britannicaURL + subpath);
    const sitemapHTML = HTMLParser.parse(await sitemapResponse.text());

    return sitemapHTML.querySelectorAll("ul.list-unstyled.md-az-browse-content")[0]
        .querySelectorAll("a")
        .map((element) => element.getAttribute("href") as string);
}

async function getArticleContentByPrint(articleContentURLPath: string): Promise<HTMLElement> {
    const dataTopicIDRegex = /data-topic-id="(\d*)"/g;

    const htmlResponse = await fetch(britannicaURL + articleContentURLPath);
    const html = await htmlResponse.text();

    const dataTopicIDString = html.matchAll(dataTopicIDRegex).next().value[1];
    const articleContentResponse = await fetch(britannicaPureDocumentURL + dataTopicIDString);
    const articleContent = await articleContentResponse.text();

    return HTMLParser.parse(articleContent);
}

async function getArticleContent(articleContentURLPath: string): Promise<string[]> {
    const url = britannicaURL + articleContentURLPath;
    const htmlResponse = await fetch(url);

    if (htmlResponse.status !== 200) {
        throw new Error(`Failed to fetch ${url} with status ${htmlResponse.status}`);
    }

    const html = HTMLParser.parse(await htmlResponse.text());

    if (html.querySelectorAll("div.topic-content").length !== 0) {
        const articleDOM = html.querySelectorAll("div.topic-content")[0];
        const paragraphsHTML = articleDOM.querySelectorAll("p.topic-paragraph");

        return paragraphsHTML.map((element: HTMLElement) => element.textContent as string);
    }

    else {
        // Select the first p.topic-paragraph element

        const paragraphHTML = html.querySelectorAll("p.topic-paragraph")[0];
        return [paragraphHTML.textContent as string];
    }
}

function simplifyArticleContent(articleDOM: HTMLElement): string {
    const articleContentHTML = articleDOM.getElementById("ref1") as HTMLElement;
    // Remove all div elements with the class "assemblies"

    articleContentHTML.querySelectorAll("div.assemblies").forEach((element) => {
        element.remove();
    });

    return convert(articleContentHTML.toString()).replace(/\[.*?\]/g, "");
}

async function getNewArticlesURL(): Promise<string[]> {
    const htmlResponse = await fetch(britannicaNewArticlesURL);
    const htmlPlaintext = await htmlResponse.text();
    const allArticleAnchors = HTMLParser.parse(htmlPlaintext).querySelectorAll("a.d-flex.p-10");

    return allArticleAnchors.map((element: HTMLElement) => element.getAttribute("href") as string);
}

function urlToPath(url: string): string {
    return url.replace(britannicaURL, "");
}

function pathToFilename(path: string): string {
    // topic/article-title becomes topic.article-title.txt

    return path.replace("/", ".") + ".txt";
}

async function main() {
    const outputDirectory = process.argv[2];

    if (!outputDirectory) {
        console.error("No output directory provided");
        process.exit(1);
    }

    const newArticlesURLs = await getNewArticlesURL();
    const newArticlesPaths = newArticlesURLs.map(urlToPath);
    const totalArticles = newArticlesPaths.length;
    var articleNumber = 0;

    for (const articlePath of newArticlesPaths) {
        (async function() {

            console.log("Pulling content for article", articlePath);
            const outputFilePath = outputDirectory + "/" + pathToFilename(articlePath);

            if (await Bun.file(outputFilePath).exists()) {
                console.log("Skipping", articlePath, "because it already exists");

                articleNumber++;
                console.log("Progress:", articleNumber, "of", totalArticles);
                return;
            }

            // const articleContent = await getArticleContentByPrint(articlePath);
            const articleContent = await getArticleContent(articlePath);

            await Bun.write(outputFilePath, articleContent);
            console.log("Completed", articlePath);
            articleNumber++;
            console.log("Progress:", articleNumber, "of", totalArticles);
        })();
    }
}

/*
const sitemapSubpaths = await allSitemapSubpaths("sitemap/" + britannicaSitemapPaths[0])
const res = await Promise.all(sitemapSubpaths.map(allArticlePathsFromSitemapSubpath))

console.log(JSON.stringify(res));

*/

/*
const result = britannicaSitemapPaths.map(async (sitemapPath) => {
    console.log("Pulling subpaths from", sitemapPath);
    const subpaths = await allSitemapSubpaths("sitemap/" + sitemapPath)
    console.log("Found", subpaths.length, "subpaths");

    const seconds = 60;

    await new Promise(resolve => setTimeout(resolve, seconds * 1000));

    return subpaths.map(async (subpath) => {
        console.log("Pulling articles from", subpath);
        const articlePaths = await allArticlePathsFromSitemapSubpath(subpath)
        console.log("Found", articlePaths.length, "articles");

        await new Promise(resolve => setTimeout(resolve, 10000));

        return articlePaths
    });
});
*/

const result: string[] = [];

for (const sitemapPath of britannicaSitemapPaths) {
    console.log("Pulling subpaths from", sitemapPath);
    const subpaths = await allSitemapSubpaths("sitemap/" + sitemapPath)
    console.log("Found", subpaths.length, "subpaths");

    for (const subpath of subpaths) {
        console.log("Pulling articles from", subpath);
        const articlePaths = await allArticlePathsFromSitemapSubpath(subpath)
        console.log("Found", articlePaths.length, "articles");

        result.push(...articlePaths);

        console.log("Waiting 10 seconds...");

        await new Promise(resolve => setTimeout(resolve, 10000));
    }

    console.log("Done pulling articles from", sitemapPath);
    console.log("Waiting 1 minute...");

    const seconds = 60;

    await new Promise(resolve => setTimeout(resolve, seconds * 1000));
}

const articleDataJSON = JSON.stringify(result);

await Bun.write("./output.json", articleDataJSON);
console.log(articleDataJSON);
