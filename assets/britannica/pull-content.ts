import HTMLParser, { HTMLElement } from "node-html-parser";
import { convert } from "html-to-text";
import { ArgumentParser } from "argparse";
import { readdirSync } from "fs";

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

    return path.replace("/", "").replace("/", ".") + ".txt";
}

function filenameToPath(filename: string): string {
    // Remove the .txt from the end if there is any and repalce . with /
    // Do the reverse of pathToFilename

    return filename.replace(".txt", "").replace(".", "/");
}

async function pullNewestArticles() {
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

async function pullAllArticlePaths() {
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
}

async function getListFromJSONFile(filePath: string): Promise<string[]> {
    const fileContent = await Bun.file(filePath).text();
    return JSON.parse(fileContent);
}

function shuffleArray<T>(array: T[]) {
    for (let index = array.length - 1; index > 0; index--) {
        const newIndex = Math.floor(Math.random() * (index + 1));
        [array[index], array[newIndex]] = [array[newIndex], array[index]];
    }

    return array;
}


async function main() {
    const parser = new ArgumentParser({
        description: "Britannica content scraper",
    });


    // Options: output file, get random article path, get random article content, get article by path

    parser.add_argument("-o", "--output", { help: "Output directory" });
    parser.add_argument("-of", "--output-file", { help: "Output filename (not reccomended)" });
    parser.add_argument("-p", "--path", { help: "Output the path and not the content" });
    parser.add_argument("-d", "--database", { help: "Specify the json article path database", required: true });
    parser.add_argument("-c", "--count", { help: "Count of articles to pull", type: Number, default: 1 });
    parser.parse_args();

    const parserResult = parser.parse_args();

    const articlePathList = await getListFromJSONFile(parserResult.database);

    if (parserResult.output) {
        // List directories in output
        const alreadyPulledArticles = readdirSync(parserResult.output);
        let notPulledArticles: string[] = []

        for (const articlePath of articlePathList) {
            const articlePathFilename = pathToFilename(articlePath);

            if (!(articlePathFilename in alreadyPulledArticles)) {
                notPulledArticles.push(articlePathFilename);
            }
        }

        // Shuffle notPulledArticles

        notPulledArticles = shuffleArray(notPulledArticles);
        const selectedNotPulledArticles = notPulledArticles.slice(0, parserResult.count);
        const selectedNotPulledArticlePaths = selectedNotPulledArticles.map(filenameToPath);

        if (parserResult.path) {
            console.log(selectedNotPulledArticlePaths.join("\n"));
        }

        else {
            for (const articlePath of selectedNotPulledArticlePaths) {
                console.log("Pulling", articlePath);
                const articleContent = await getArticleContent(britannicaURL + articlePath);
                console.log("Writing", articlePath);
                await Bun.write(parserResult.output + "/" + pathToFilename(articlePath), articleContent.join("\n"));
            }
        }
    }

    else if (parserResult.output_file) {
    }

    else {
        // Print to stdout
    }
}

await main();

