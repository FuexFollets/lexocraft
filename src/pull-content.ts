import * as wiki from "wikipedia";
import { convert } from "html-to-text";

const jsEngine = process.argv.at(0);
const scriptPath = process.argv.at(1);

if (process.argv.length <= 2) {
    const usage = `
    Usage: ${jsEngine} ${scriptPath} <options>

    Options: get [URL: string], random [N (count): number], search [query: string]
    `

    console.log(usage);

    process.exit(0);
}

const option = process.argv.at(2);

function simplifyHTMLtoText(html: string): string[] {
    const text = convert(html).replace(/\[.*?\]/g, "");

    const sectionSeparator: string = "\n\n--------------------------------------------------------------------------------\n\n";

    const sections = text.split(sectionSeparator);

    // Section: /w/index.php?title=David_S._Lewis&action=edit&section=0
    // File: ./File:МПГВ_КШИ_9_на_Красной_площади_в_Параде_Победы_9_мая_2015_г.jpgCadets
    /* Quick facts:
Quick Facts Born, Died ...
Charles McElroy White
./Born(1891-06-13)June 13,
1891

Oakland, Maryland , U.S.
DiedJanuary 10, 1977(1977-01-10) (aged 85)

Palm Beach, Florida , U.S.
Occupation(s)President, Republic Steel  (1945–1956);
CEO and Chairman, Republic Steel (1956–1960)Years active1927–1966SpouseHelen
Bradley White

Close
*/

    /* More information:
More information Review scores, Source ...

Professional ratingsReview scoresSourceRatingAllMusic


Close

*/

    // Dot slash prefix: ./2023_Formula_Drift_season#pcs-ref-back-link-cite_note-19

    // List Item: * Rudee Lipscomb as Marcy

    const sectionRegex = /\/w\/index\.php\?title=.*?&action=edit&section=\d+/g;
    const fileRegex = /File:.*?\.(jpg|svg)/g;

    // Quick facts start with "Quick Facts" and end with "Close"

    const quickFactsRegex = /Quick Facts([\s\S]*?)Close/g;

    // More information start with "More information" and end with "Close"

    const moreInformationRegex = /More information([\s\S]*?)Close/g;

    // Any sequence of characters that starts with a dot and a slash unterminated by any type of whitespace
    const dotSlashPrefixRegex = /\.\/.*?\s/g;

    const listItemRegex = /\*.*?\n/g;

    sections.forEach((section, index) => {
        sections[index] = section
            .replace(sectionRegex, "")
            .replace(fileRegex, "")
            .replace(quickFactsRegex, "")
            .replace(moreInformationRegex, "")
            .replace(dotSlashPrefixRegex, "")
            .replace(listItemRegex, "");
    });

    return sections;
}

function isDisambiguationPage(text: string): boolean {
    const disambiguationPageTerm = "Topics referred to by the same term";

    return text.includes(disambiguationPageTerm);

}

function get(title: string): Promise<string | wiki.notFound> {
    return wiki.mobileHtml(title);
}

async function randomWikiTitle(count: number): Promise<string[]> {
    const wikipediaRandom = "https://en.wikipedia.org/wiki/Special:Random";
    const wikipediaRandomAPIEndpoint = `https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=${count}&format=json`;

    // https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=1&format=json
    const responses: Response[] = [];

    for (let iteration = 0; iteration < count; iteration++) {
        responses.push(await fetch(wikipediaRandomAPIEndpoint));
    }

    const titles: string[] = [];

    for (const response of responses) {
        const json = await response.json() as any;

        titles.push(json.query.random.at(0).title);
    }

    return titles;
}

async function getRandomWikiArticle(): Promise<string | wiki.notFound> {
    const articleName = (await randomWikiTitle(1)).at(0) as string;
    return await get(articleName);

}

async function randomPassage(minimumLength: number): Promise<string> {
    while (true) {
        const article = await getRandomWikiArticle();

        const articleText = simplifyHTMLtoText(article as string)[1];

        if (articleText.length >= minimumLength) {
            return articleText;
        }
    }
}

console.log(await randomPassage(1000));
