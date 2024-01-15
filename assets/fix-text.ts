async function fixFile(filename: string) {
    const emdash = "—";
    const rightDoubleQuotationMark = "”";
    const leftDoubleQuotationMark = "“";

    const file = Bun.file(filename);
    const text = await file.text();

    const fixedText = text.replaceAll(emdash, "-")
        .replaceAll(rightDoubleQuotationMark, '"')
        .replaceAll(leftDoubleQuotationMark, '"');

    Bun.write(file, fixedText);
}

async function main() {
    const files = process.argv.slice(2);
    console.log("Fixing files:", files);

    if (files.length === 0) {
        console.error("No files provided");
        process.exit(1);
    }

    var count = 0;

    for (const file of files) {
        (async function() {
            const thisCount = ++count;
            console.log("Fixing file", file, "(", thisCount, "of", files.length, ")");
            fixFile(file);
            console.log("Done fixing file", file, "(", thisCount, "of", files.length, ")");
        })();
    }
}

await main();
