all:
	pdflatex finalpaper.tex
	biber finalpaper
	pdflatex finalpaper.tex

