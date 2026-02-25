"""
Publication Export: Generates LaTeX reports for discovery results.
"""
import sympy as sp

def generate_latex_report(results_list, save_path="results/report.tex"):
    """
    Takes a results list (from Pareto analysis) and creates a LaTeX table.
    """
    latex_template = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}

\begin{titlepage}
    \centering
    \scshape\huge Scientific Discovery Report \\
    \vspace{1cm}
    \scshape\Large Algorithm Evolver Engine \\
    \vspace{2cm}
\end{titlepage}

\begin{document}

\section{Pareto Front Solutions}
The following table summarizes the most physically significant solutions found.

\begin{table}[h!]
\centering
\begin{tabular}{lrrp{8cm}}
\toprule
\textbf{Complexity} & \textbf{$\chi^2$ (Norm)} & \textbf{LaTeX Formula} \\
\midrule
%ROWS%
\bottomrule
\end{tabular}
\caption{Discovery results sorted by complexity.}
\end{table}

\end{document}
"""
    
    rows = []
    for res in results_list:
        try:
            # Convert simplified string to LaTeX
            expr = sp.sympify(res['simplified'])
            latex_formula = f"${sp.latex(expr)}$"
            
            row = f"{res['complexity']:.1f} & {res['mse']:.4f} & {latex_formula} \\\\"
            rows.append(row)
        except Exception:
            continue
            
    final_latex = latex_template.replace("%ROWS%", "\n".join(rows))
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(final_latex)
    
    return save_path

if __name__ == "__main__":
    # Test export
    mock_results = [
        {'complexity': 7.0, 'mse': 98.4, 'simplified': 'v**2 * sin(2*angle) / 9.81'},
        {'complexity': 3.0, 'mse': 500.0, 'simplified': 'v * angle'}
    ]
    path = generate_latex_report(mock_results)
    print(f"Report generated at {path}")
