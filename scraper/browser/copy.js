async page => {
  await page.bringToFront();
  await page.context().grantPermissions(['clipboard-read','clipboard-write'], {origin:'https://github.com'});
  const sentinel = 'github-copy-pending-' + Date.now() + '-' + Math.random();
  await page.evaluate(s => navigator.clipboard.writeText(s), sentinel);
  await page.getByRole('button', {name:'Copy raw file',exact:true}).click({timeout:15000});
  // waitForFunction polls synchronous predicates; explicitly await clipboard reads.
  let copied = sentinel;
  for (let attempt=0;attempt<150 && copied===sentinel;attempt++) {
    copied = await page.evaluate(() => navigator.clipboard.readText());
    if (copied===sentinel) await page.waitForTimeout(100);
  }
  if (copied===sentinel) throw Error('Copy raw file did not update the clipboard');
  return await page.evaluate(copied => {
    const bytes = new TextEncoder().encode(copied);
    // Keep a page-local snapshot so subsequent chunks do not touch the clipboard.
    let binary = '';
    for (let i=0;i<bytes.length;i+=8192) binary += String.fromCharCode(...bytes.subarray(i,i+8192));
    window.__githubCopyBytes = btoa(binary);
    return {bytes:bytes.length, encoded:window.__githubCopyBytes.length};
  }, copied);
}
