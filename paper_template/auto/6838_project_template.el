(TeX-add-style-hook
 "6838_project_template"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("graphicx" "pdftex" "dvips")))
   (TeX-run-style-hooks
    "latex2e"
    "6838publ"
    "6838publ10"
    "6838"
    "graphicx"
    "t1enc"
    "dfadobe"
    "egweblnk"
    "cite"
    "lipsum"
    "amsmath"
    "amsfonts")
   (LaTeX-add-labels
    "fig:teaser"
    "sec:intro"
    "sec:related_work"
    "eq:naive_energy"
    "sec:technical_approach"
    "eq:frame_equation"
    "fig:frame_equation"
    "eq:edge_equation"
    "fig:edge_equation"
    "eq:frame_energy"
    "eq:edge_energy"
    "eq:sum_energy"
    "fig:original"
    "fig:rotated"
    "fig:original_handle"
    "fig:stretched_handle"
    "fig:n50soft"
    "fig:n70soft"
    "fig:n70rigid"
    "fig:step_mapping"
    "fig:low_weight")
   (LaTeX-add-bibliographies
    "6838bibsample"))
 :latex)

