const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

async function exportPdf() {
  const input = path.resolve(process.argv[2] || path.join(__dirname, '..', 'example', 'slides.html'));
  const output = path.resolve(process.argv[3] || input.replace(/\.html$/i, '.pdf'));
  if (input === output) throw new Error('Укажите разные пути для HTML и PDF.');
  await fs.access(input);

  const executablePath = process.env.CHROME_EXECUTABLE;
  const browser = await chromium.launch(executablePath ? {executablePath} : {});
  try {
    const page = await browser.newPage({offline: true, viewport: {width: 1280, height: 720}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => errors.push(new URL(request.url()).protocol + ' resource failed'));
    const url = pathToFileURL(input);
    url.searchParams.set('print-pdf', '');
    await page.goto(url.href);
    await page.waitForFunction(() => window.Reveal?.isReady());
    const hasMath = await page.locator('.slides .math').count() > 0;
    if (hasMath) {
      await page.waitForFunction(() => window.MathJax?.startup?.promise);
      await page.evaluate(() => window.MathJax.startup.promise);
    }
    await page.evaluate(() => document.fonts.ready);
    // Print preparation temporarily reparents sections, so derive the expected
    // count from the rendered source and wait for that complete PDF DOM.
    const source = await fs.readFile(input, 'utf8');
    const count = (source.match(/<section[^>]+class="[^"]*\bslide\b/g) || []).length;
    if (count < 1) throw new Error('No slide sections found; refusing an empty PDF.');
    await page.waitForFunction(total => document.querySelectorAll('.pdf-page').length === total, count);

    const broken = await page.evaluate(() => ({
      images: [...document.querySelectorAll('.pdf-page img')].filter(img => !img.complete || !img.naturalWidth).length,
      formulas: document.querySelectorAll('mjx-container [data-mml-node=merror], [data-mjx-error]').length,
    }));
    if (broken.images || broken.formulas) {
      throw new Error('В печатной версии есть ошибки в изображениях или формулах.');
    }

    const pdf = await page.pdf({
      preferCSSPageSize: true,
      printBackground: true,
      displayHeaderFooter: false,
      tagged: true,
    });
    if (errors.length) throw new Error(errors.join('\n'));
    await fs.writeFile(output, pdf);
    console.log('PDF собран. Слайдов ' + count + '. Файл ' + output);
  } finally {
    await browser.close();
  }
}

exportPdf().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});

