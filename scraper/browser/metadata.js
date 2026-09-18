async page => {
  await page.waitForFunction(() => document.querySelector('script[data-target="react-app.embeddedData"]'), {}, {timeout: 20000});
  return await page.evaluate(() => {
    const p = JSON.parse(document.querySelector('script[data-target="react-app.embeddedData"]').textContent).payload;
    const r = p.codeViewTreeRoute || p.codeViewRepoRoute || p.codeViewBlobLayoutRoute;
    if (!r) throw Error('Unsupported GitHub page, authentication required, or empty repository');
    const b = r.blob;
    return {path:r.path === '/' ? '' : r.path, commit:r.refInfo.currentOid, tree:r.tree && {items:r.tree.items,totalCount:r.tree.totalCount},
      blob:b && {hash:b.headerInfo.shortPath, lfs:b.headerInfo.isGitLfs, mode:b.headerInfo.mode, binary:b.image || !b.viewable}};
  });
}
