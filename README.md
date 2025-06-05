## Content for my Personal Website

This repository contains all the content for local testing. Actual deployed website can be found [here](https://fquico1999.github.io/).

This is based on the [Academic Theme](https://sourcethemes.com/academic/) for [Hugo](https://gohugo.io/)

## Updating the website

The site is generated with [Hugo](https://gohugo.io/) using the Academic theme.
To pull in the latest theme updates and regenerate the public site:

1. Update the theme submodule:

   ```bash
   bash update_academic.sh
   ```

2. Preview changes locally:

   ```bash
   ./view.sh
   ```

3. Build and deploy the site:

   ```bash
   ./deploy.sh
   ```

The `deploy.sh` script builds the site with Hugo and pushes the contents of
`public/` to the repository.
