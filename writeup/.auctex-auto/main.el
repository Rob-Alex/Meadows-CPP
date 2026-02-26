;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "-shell-escape")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "lettersize" "journal")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "") ("amsfonts" "") ("algorithmic" "") ("algorithm" "") ("array" "") ("subfig" "caption=false" "font=normalsize" "labelfont=sf" "textfont=sf") ("textcomp" "") ("stfloats" "") ("url" "") ("verbatim" "") ("graphicx" "") ("caption" "") ("subcaption" "") ("orcidlink" "") ("cite" "") ("hyperref" "") ("subfiles" "")))
   (TeX-run-style-hooks
    "latex2e"
    "content"
    "IEEEtran"
    "IEEEtran10"
    "amsmath"
    "amsfonts"
    "algorithmic"
    "algorithm"
    "array"
    "subfig"
    "textcomp"
    "stfloats"
    "url"
    "verbatim"
    "graphicx"
    "caption"
    "subcaption"
    "orcidlink"
    "cite"
    "hyperref"
    "subfiles")
   (TeX-add-symbols
    "BibTeX"))
 :latex)

