# UltraEP Blog Preview

This directory contains the minimal Jekyll site for the UltraEP blog. The public routes are:

- `/` serves the English blog
- `/en/` redirects to `/`
- `/zh/` serves the Chinese blog

## Local Preview

Use Homebrew Ruby instead of the macOS system Ruby:

```bash
cd docs
export PATH="$(brew --prefix ruby)/bin:$PATH"
bundle config set path vendor/bundle
bundle update
bundle exec jekyll serve --host 127.0.0.1 --port 4000 --baseurl ""
```

Open `http://127.0.0.1:4000/`.

VSCode Markdown preview can show the article images directly because the image paths are relative to this directory. The top navigation, theme toggle, and pretty URLs are Jekyll layout features, so use the local server above to preview the deployed page.
