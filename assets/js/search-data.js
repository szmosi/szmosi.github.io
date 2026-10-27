// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "Projects",
          description: "A growing collection of my Data Science projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "projects-nlp-election-analytics-classical-vs-transformer-models",
          title: 'NLP Election Analytics: Classical vs Transformer Models',
          description: "An MSc group project comparing sentiment analysis and topic modelling methods on 2024 US election-related Twitter/X discussions.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/nlp_election_analytics/";
            },},{id: "projects-python-speed-lab",
          title: 'Python Speed Lab',
          description: "Benchmarking CPU vs. GPU performance across NumPy, Numba, Cython, and CUDA.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/python_speed_lab/";
            },},{id: "projects-road-damage-classification",
          title: 'Road Damage Classification',
          description: "Comparing SVM and convolutional neural networks for image-based infrastructure inspection.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/road_damage_classification/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
