const britannicaURL: string = "https://www.britannica.com/";
const britannicaPureDocumentURL: string = "https://www.britannica.com/print/article/";

async function getArticleContent(articleContentURLPath: string) {
    const dataTopicIDRegex = /data-topic-id="(\d*)"/g;

    const htmlResponse = await fetch(britannicaURL + articleContentURLPath);
    const html = await htmlResponse.text();

    const dataTopicIDString = html.matchAll(dataTopicIDRegex).next().value[1];
    const articleContentResponse = await fetch(britannicaPureDocumentURL + dataTopicIDString);
    const articleContent = await articleContentResponse.text();

    console.log(articleContent);
}

getArticleContent("topic/sancocho");
