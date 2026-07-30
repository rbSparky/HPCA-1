PAPER=main
LATEX=pdflatex
FLAGS=-interaction=nonstopmode -halt-on-error

.PHONY: all clean
all:
	$(LATEX) $(FLAGS) $(PAPER).tex
	$(LATEX) $(FLAGS) $(PAPER).tex
	$(LATEX) $(FLAGS) $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).pdf
