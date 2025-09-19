import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



# Verschiedene Werte für die Anzahl von Stützpunkten
N_values = [8,16,32,64,128, 256, 512, 1024, 2048]
# Konstante für die Größe der Plots 
FIGSIZE = (16,10)



def generiere_tridiagonale_matrix_lin(b,d,f,alpha, beta, N):
    # EINGABE: Funktionen für b,d,f und Konstanten alpha und beta, N Anzahl der Stützpunkte
    # AUSGABE Haupt-, obere-und untere Diagonalen einer Tridiagonalmatrix und die rechte Seite eines linearen Gleichungssystems

    # Schrittweite
    h = 1/(N-1)

    # Werte der Tridiagonalmatrix initialisieren 
    main_diagonal = (2/h**2)*np.ones(N-2) 
    upper_diagonal = ((-1)/(h**2))*np.ones(N-3) 
    lower_diagonal = ((-1)/(h**2))*np.ones(N-3)  
    right_hand_side = np.zeros(N-2) 

    # Gitter erstellen 
    x = np.linspace(0, 1, N)
    
    # Auffüllen 
    for i in range(1,N-1):
        d_value = d(x[i])
        b_value = b(x[i])

        main_diagonal[i-1] += d_value
        
        if i < N-2:
            upper_diagonal[i-1] += (b_value)/(2*h)

        if i > 1:
            lower_diagonal[i-2] -= (b_value)/(2*h)
    
        # Rechte Seite 
        # Erster Eintrag 
        if i == 0:
            right_hand_side[i-1] = f(x[i]) + alpha*((1/(h**2))+((b_value)/(2*h)))
        # Letzter Eintrag
        elif i == N-2:
            right_hand_side[i-1] = f(x[i]) + beta*((1/(h**2))+(b_value/(2*h)))
        else: 
            right_hand_side[i-1] = f(x[i])
    
    return main_diagonal, upper_diagonal, lower_diagonal, right_hand_side



def thomas(d,f,e,b):
    # Implementiert den Thomas-Algorithmus (vgl. 6.3.5)
    # EINGABE: d,f, und b sind Arrays mit Werten der Tridiagonalmatrix (vgl. 6.3.5) 
    # AUSGABE: Lösungsvektor als Array des Gleichungssystems
    
    #Größe der nxn Matrix
    n = len(d)
    # Wird später unsere Lösungen enthalten
    X = np.zeros(n)

    # S1. Elimination
    for i in range(1,n): 
        l = e[i-1]/d[i-1]
        d[i] = d[i] - l*f[i-1]
        b[i] = b[i] - l*b[i-1]

    #S1 Rückwärtsersetzung
    X[-1] = b[-1]/d[-1]
    for i in range(n-2,-1,-1):
        X[i] = (b[i] - f[i]*X[i+1]) / d[i]

    return X



def extrapolate(U1, U2, N1, N2, alpha, beta, k):
    # Implementiert einen Schritt der Richardson Extrapolation (vgl. Algorithmus 5.6.2)
    # EINGABE: Listen U1 und U2 mit Werten der approximierten Lösung für verschiedene 
    #          Anzahl Stützpunkte N1, N2, Konstanten alpha und beta, k ist die Konvergenzrate 
    # AUSGABE: Neue approximierte Werte der Lösungswerte gemäß Algorithmus 5.6.2

    factor = (1)/((N2/N1)**k - 1)
    # Enthält Lösungen später
    extrapoliertes_U2 = []

    for i in range(0,len(U2), 2):
    # Durchläuft jeden zweiten Wert von U2, da nur dort die Stützpunkte 
    # von U1 und U2 übereinstimmen 
        extrapoliertes_U2.append(U2[i]+ factor*(U2[i] - U1[int(i/2)])) 
    
    return extrapoliertes_U2



def Richardson_Extrapolation(u, N, alpha, beta, k=1):
    # Führt das Verfahren der Richardson Extrapolation (vgl. Abschnitt 5.6) durch
    # EINGABE: Liste u ist ein Array mit Listen von Lösungswerten für die verschiedenen Gitterweiten 
    #          N ist die Liste mit verschiedenen Werten für die Anzahl von Stützpunkten
    # AUSGABE: Liste mit neuen extrapolierten Werten in den Stützpunkten

    neues_u = []
    
    for i in range(len(N)-1):
        U1 = u[i]
        U2 = u[i+1]
        
        N1 = N[i]
        N2 = N[i+1]

        neues_u.append(extrapolate(U1, U2, N1, N2, alpha, beta, k))

    return neues_u




# LINEARER FALL
# -----------------------------------------------------------------------
# TESTAUFGABE 1: -u''(x) + u'(x) + 2u(x) = -6 

def b_lin(x):
    return 1

def d_lin(x):
    return 2

def f_lin(x):
    return -6

# Konstruierte Lösung
def solution_u_lin(x):
    return (3*np.exp(2*x))/(np.exp(2)+np.exp(1)+1) + (3*np.exp(-x)*np.exp(1)*(np.exp(1)+1))/(np.exp(2)+np.exp(1)+1) - 3 


alpha = solution_u_lin(0)
beta = solution_u_lin(1)


# Für die Extrapolation später
solutions = [] 

# Größe des Plots wird festgelegt 
plt.figure(figsize=FIGSIZE)

for N in N_values:
    m,u,l,r = generiere_tridiagonale_matrix_lin(b_lin,d_lin,f_lin,alpha,beta, N+1)
    numerical_solution = thomas(m,u,l,r)
    # Es werden nur die inneren Werte des Vektors u bestimmt. Man muss also noch 
    # die Konstanten alpha und beta an den Anfang bzw. Ende setzen. 
    numerical_solution_final = np.concatenate(([alpha], numerical_solution, [beta]))
    # Lösung wird in die Liste aufgenommen 
    solutions.append(numerical_solution_final)

    # Stützpunkte für numerische Lösung erstellen 
    x_numeric = np.linspace(0, 1, N+1)

    # Numerische Lösung plotten
    plt.plot(x_numeric, numerical_solution_final, label=f'Numerische Lösung (N={N})', marker='o', linestyle='--')
    
# Analytische Lösung plotten für das zuletzt bestimmte Gitter 
plt.plot(x_numeric, solution_u_lin(x_numeric), label='Analytische Lösung', color='black', linewidth=2)


# Plot formatieren 
plt.title('Vergleich Numerische und Analytische Lösung (Linearer Fall 1)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid()
plt.show()




# ERROR ANALYSE
# -----------------------------------------------------------------------

max_errors = []

plt.figure(figsize=FIGSIZE)

# Berechnet den Fehler und plottet Ergebnisse 
for N in N_values:
    m,u,l,r = generiere_tridiagonale_matrix_lin(b_lin,d_lin,f_lin,alpha,beta, N+1)
    numerical_solution = thomas(m,u,l,r)
    numerical_solution_final = np.concatenate(([alpha], numerical_solution, [beta]))

    x_numeric = np.linspace(0, 1, N+1)
    # Konstruierte Lösung an Stützpunkten berechnen 
    analytical_solution = solution_u_lin(x_numeric) 

    # Absoluten Fehler berechnen 
    error = np.abs(analytical_solution - numerical_solution_final)
    # Davon den maximalen Wert bestimmen und in Liste aufnehmen
    max_error_momentan = np.max(error)
    max_errors.append(max_error_momentan)



plt.plot(N_values, max_errors, marker='o', linestyle='-', color='blue', markersize=8, label='Globaler Max Error')


# Plot formatieren
# Doppelt logarithmische Darstellung 
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error vs. N (Linearer Fall 1)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()
plt.show()

# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = {
        "N": N_values,
        "Max Error": max_errors,
        }

df = pd.DataFrame(data)

csv_file = "error_table_ohne_extrapolation_linear_1.csv"
df.to_csv(csv_file, index=False)

print("Linearer Fall 1")
print("----------------------------------------")
print()
# Tabelle ausgeben 
print()
print()
print(df.to_string(index=False))
print()
print()




# EXTRAPOLATION
#--------------------------------------------------------------------------

# Erster Durchlauf 
Extrapolierte_solution_1 = Richardson_Extrapolation(solutions, N_values, alpha, beta, 2) 
# Zweiter Durchlauf
Extrapolierte_solution_2 = Richardson_Extrapolation(Extrapolierte_solution_1, N_values[0:-1], alpha, beta, 4)



i = 0
plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:
    x_numeric = np.linspace(0, 1, N + 1)
    # Die extrapolierten Lösungen für die verschiedenen N plotten 
    plt.plot(x_numeric, Extrapolierte_solution_1[i], label=f'Extrapolierte Lösung (N={N})', marker='o', linestyle='--')
    i += 1

# Zum Vergleich, die analytische Lösung für das zuletzt bestimmte Gitter plotten 
plt.plot(x_numeric, solution_u_lin(x_numeric), label='Analytische Lösung', color='black', linewidth=2)

plt.title('Vergleich Extrapolierte und Analytische Lösung (Linearer Fall 1)')
plt.xlabel('x')
plt.ylabel('Extrapoliertes u(x)')
plt.legend()
plt.grid()
plt.show()

i = 0
# Enthält später die maximalen Fehler nach einem Extrapolationsdurchlauf 
max_errors_extrapolation_1 = []

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:

    x_numeric = np.linspace(0,1,N+1)
    # Analytische Lösung in den Stützpunkten berechnen 
    analytical_solution = solution_u_lin(x_numeric) 
    
    # Absoluter Fehler 
    error = np.abs(analytical_solution - Extrapolierte_solution_1[i])
    # Davon den maximalen Wert bestimmen und die Liste aufnehmen
    max_error_momentan = np.max(error)
    max_errors_extrapolation_1.append(max_error_momentan)
    i += 1



i = 0
# Enthält später die maximalen Fehler nach dem zweiten Extrapolationsdurchlauf 
max_errors_extrapolation_2 = []

for N in N_values[0:-2]:

    x_numeric = np.linspace(0,1,N+1)
    analytical_solution = solution_u_lin(x_numeric)
    
    error = np.abs(analytical_solution - Extrapolierte_solution_2[i])
    max_error_momentan = np.max(error)
    max_errors_extrapolation_2.append(max_error_momentan)
    i += 1


# Maximalen Fehler für Verfahren ohne Extrapolation, nach einem Durchlauf und nach dem zweiten plotten 
plt.plot(N_values[0:-2], max_errors[0:-2], marker='o', linestyle='-', color='blue', markersize=8, label='Ohne Extrapolation')
plt.plot(N_values[0:-2], max_errors_extrapolation_1[0:-1], marker='o', linestyle='-', color='orange', markersize=8, label='Mit Extrapolation (1. Durchlauf)')
plt.plot(N_values[0:-2], max_errors_extrapolation_2, marker='o', linestyle='-', color='red', markersize=8, label='Mit Extrapolation (2. Durchlauf)')


#Plot formatieren
# Doppelt logarithmische Darstellung
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error ohne und mit Extrapolation vs. N (Linearer Fall 1)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()
plt.show()


# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = { 
        "N": N_values[0:-2],
        "Max Error (Ohne Extrapolation)": max_errors[0:-2],
        "Max Error (Mit Extrapolation (1. Durchlauf))": max_errors_extrapolation_1[0:-1],
        "Max Error (Mit Extrapolation (2. Durchlauf))": max_errors_extrapolation_2,
    }

df = pd.DataFrame(data)

csv_file = "error_table_mit_Extrapolation_linear_1.csv"
df.to_csv(csv_file, index=False)


print("Linearer Fall 1 nach Extrapolation")
print("----------------------------------------")
print()
# Tabelle ausgeben 
print()
print()
print(df.to_string(index=False))
print()
print()



# ------------------------------------------------------------------------------------
# TESTAUFGABE 2: -u''(x) + 2u'(x) + u(x) = 1/(4*x^(3/2)) + 1/x^(1/2) + x^(1/2)


def b_lin_2(x):
    return 2

def d_lin_2(x):
    return 1

def f_lin_2(x):
    return (1)/(4*(x**(3/2))) + (1)/(x**(1/2)) + x**(1/2)

# Konstruierte Lösung
def solution_u_lin_2(x):
    return x**(1/2)


alpha = solution_u_lin_2(0)
beta = solution_u_lin_2(1)

# Für die Extrapolation später
solutions_2 = [] 

plt.figure(figsize=FIGSIZE)

for N in N_values:
    m,u,l,r = generiere_tridiagonale_matrix_lin(b_lin_2,d_lin_2,f_lin_2,alpha,beta, N+1)
    numerical_solution = thomas(m,u,l,r)
    # Es werden nur die inneren Werte des Vektors u bestimmt. Man muss also noch 
    # die Konstanten alpha und beta an den Anfang bzw. Ende setzen. 
    numerical_solution_final = np.concatenate(([alpha], numerical_solution, [beta]))
    # Lösung wird in die Liste aufgenommen 
    solutions_2.append(numerical_solution_final)

    # Stützpunkte für numerische Lösung erstellen 
    x_numeric = np.linspace(0, 1, N+1)

    # Numerische Lösung plotten
    plt.plot(x_numeric, numerical_solution_final, label=f'Numerische Lösung (N={N})', marker='o', linestyle='--')
    
# Analytische Lösung plotten 
plt.plot(x_numeric, solution_u_lin_2(x_numeric), label='Analytische Lösung', color='black', linewidth=2)


# Plot formatieren 
plt.title('Vergleich Numerische und Analytische Lösung (Linearer Fall 2)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid()
plt.show()




# ERROR ANALYSE
# -----------------------------------------------------------------------

max_errors = []

plt.figure(figsize=FIGSIZE)

# Berechnet den Fehler und plottet Ergebnisse 
for N in N_values:
    m,u,l,r = generiere_tridiagonale_matrix_lin(b_lin_2,d_lin_2,f_lin_2,alpha,beta, N+1)
    numerical_solution = thomas(m,u,l,r)
    numerical_solution_final = np.concatenate(([alpha], numerical_solution, [beta]))

    x_numeric = np.linspace(0, 1, N+1)
    # Konstruierte Lösung an Stützpunkten berechnen 
    analytical_solution = solution_u_lin_2(x_numeric)

    # Absoluten Fehler berechnen 
    error = np.abs(analytical_solution - numerical_solution_final)
    # Davon den maximalen Wert bestimmen und in die Liste aufnehmen
    max_error_momentan = np.max(error)
    max_errors.append(max_error_momentan)



plt.plot(N_values, max_errors, marker='o', linestyle='-', color='blue', markersize=8, label='Globaler Max Error')


# Plot formatieren
# Doppelt logarithmische Darstellung 
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error vs. N (Linearer Fall 2)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()
plt.show()


# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = {
        "N": N_values,
        "Max Error": max_errors,
        }

df = pd.DataFrame(data)

csv_file = "error_table_ohne_extrapolation_linear_2.csv"
df.to_csv(csv_file, index=False)


print("Linearer Fall 2")
print("----------------------------------------")
print()
# Tabelle ausgeben 
print()
print()
print(df.to_string(index=False))
print()
print()




# EXTRAPOLATION
#--------------------------------------------------------------------------

# Erster Durchlauf 
Extrapolierte_solution_1_2 = Richardson_Extrapolation(solutions_2, N_values, alpha, beta, 2)
# Zweiter Durchlauf
Extrapolierte_solution_2_2 = Richardson_Extrapolation(Extrapolierte_solution_1_2, N_values[0:-1], alpha, beta, 4)



i = 0

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:
    x_numeric = np.linspace(0, 1, N + 1)
    plt.plot(x_numeric, Extrapolierte_solution_1_2[i], label=f'Extrapolierte Lösung (N={N})', marker='o', linestyle='--')
    i += 1


plt.plot(x_numeric, solution_u_lin_2(x_numeric), label='Analytische Lösung', color='black', linewidth=2)

plt.title('Vergleich Extrapolierte und Analytische Lösung (Linearer Fall 1)')
plt.xlabel('x')
plt.ylabel('Extrapoliertes u(x)')
plt.legend()
plt.grid()
plt.show()

i = 0
# Enthält später die maximalen Fehler nach einem Extrapolationsdurchlauf 
max_errors_extrapolation_1_2 = []

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:

    x_numeric = np.linspace(0,1,N+1)
    # Konstruierte Lösung an Stützpunkten bestimmen 
    analytical_solution = solution_u_lin_2(x_numeric) 
    
    # Absoluten Fehler berechnen
    error = np.abs(analytical_solution - Extrapolierte_solution_1_2[i])
    # Davon den maximalen Wert bestimmen und in die Liste aufnehmen
    max_error_momentan = np.max(error)
    max_errors_extrapolation_1_2.append(max_error_momentan)
    i += 1



i = 0
# Enthält später die maximalen Fehler nach dem zweiten Extrapolationsdurchlauf 
max_errors_extrapolation_2_2 = []

for N in N_values[0:-2]:

    x_numeric = np.linspace(0,1,N+1)
    # Konstruierte Lösung an Stützpunkten bestimmen 
    analytical_solution = solution_u_lin_2(x_numeric) 
    
    error = np.abs(analytical_solution - Extrapolierte_solution_2_2[i])
    max_error_momentan = np.max(error)
    max_errors_extrapolation_2_2.append(max_error_momentan)
    i += 1


# Maximalen Fehler für Verfahren ohne Extrapolation, nach einem Durchlauf und nach dem zweiten plotten 
plt.plot(N_values[0:-2], max_errors[0:-2], marker='o', linestyle='-', color='blue', markersize=8, label='Ohne Extrapolation')
plt.plot(N_values[0:-2], max_errors_extrapolation_1_2[0:-1], marker='o', linestyle='-', color='orange', markersize=8, label='Mit Extrapolation (1. Durchlauf)')
plt.plot(N_values[0:-2], max_errors_extrapolation_2_2, marker='o', linestyle='-', color='red', markersize=8, label='Mit Extrapolation (2. Durchlauf)')


#Plot formatieren
# Doppelt logarithmische Darstellung
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error ohne und mit Extrapolation vs. N (Linearer Fall 1)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()
plt.show()



# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = { 
        "N": N_values[0:-2],
        "Max Error (Ohne Extrapolation)": max_errors[0:-2],
        "Max Error (Mit Extrapolation (1. Durchlauf))": max_errors_extrapolation_1_2[0:-1],
        "Max Error (Mit Extrapolation (2. Durchlauf))": max_errors_extrapolation_2_2,
    }

df = pd.DataFrame(data)

csv_file = "error_table_mit_Extrapolation_linear_2.csv"
df.to_csv(csv_file, index=False)


print("Linearer Fall 2 nach Extrapolation")
print("----------------------------------------")
print()
# Tabelle ausgeben
print()
print()
print(df.to_string(index=False))
print()
print()

# ----------------------------------------------------------------------




def generiere_tridiagonale_jacobi_matrix(b, dc, c, u, f, N):
    # Generiert Haupt-,obere-und untere-Diagonalen einer Tridiagonalmatrix
    # Eingabe: Funktionen b und f, Ableitung von c nach u, Array u mit momentanen Werten, N Anzahl der Stützpunkte

    # Schrittweite
    h = 1/(N-1)
    
    # Werte initialisieren
    main_diagonal = (2/(h**2))*np.ones(N-2)
    upper_diagonal = ((-1)/(h**2))*np.ones(N-3)
    lower_diagonal = ((-1)/(h**2))*np.ones(N-3)
    right_hand_side = np.zeros(N-2)

    # Gitter erstellen 
    x = np.linspace(0, 1, N)
    
    # Auffüllen
    for i in range(1,N-1):
        b_value = b(x[i])
        dc_value = dc(x[i],u[i])
        c_value = c(x[i], u[i])

        main_diagonal[i-1] += dc_value

        if i < N-2:
            upper_diagonal[i-1] += (b_value)/(2*h)

        if i > 1:
            lower_diagonal[i-2] -= (b_value)/(2*h)
        
        # Erster Eintrag
        if i == 1:
            right_hand_side[i-1] = f(alpha,u[i],u[i+1], x[i], h, b_value, c_value)
        # Letzter Eintrag
        elif i == N-2:
            right_hand_side[i-1] = f(u[i-1],u[i],beta, x[i], h, b_value, c_value) 
        else: 
            right_hand_side[i-1] = f(u[i-1],u[i],u[i+1], x[i], h, b_value, c_value) 

    return main_diagonal, upper_diagonal, lower_diagonal, right_hand_side



def newton(b, dc, c, u_n, f, N, alpha, beta, tol=1e-6, maxiter=1000):
    # Implementiert das Newton Verfahren (vgl. Numerik 1, Abschnitt 2.3)
    # Eingabe: Funktionen b,f und c, Ableitung dc von c nach u, Anzahl der Stützpunkte N, Konstanten alpha und Beta
    for i in range(maxiter):
        m, up, l, r = generiere_tridiagonale_jacobi_matrix(b,dc, c, u_n, f, N)

        # Mit Thomas Algorithmus lösen
        u_delta = thomas(m,up,l,-r)
        u_delta_final = np.concatenate(([0], u_delta, [0]))

        # Updaten
        u_n += u_delta_final

        # Nachschauen ob schon konvergiert wurde
        if np.linalg.norm(u_delta, ord=np.inf) < tol:
            print(f"{N-1}: Konvergierte nach {i+1} Iterationen.\n")
            return u_n
        
    print(f"{N-1}: Konvergierte nicht unter {maxiter} Iterationen.\n")
    return u_n
    




# Allgemeiner Fall
# ---------------------------------------------
# TESTAUFGABE 1: -u''(x) + (exp(x)+1)u'(x) + u^2(x) = 0

def b_alg(x):
    return np.exp(x) + 1

def c_alg(x,u):
    return u**2

def dc(x,u):
    return 2*u

def f_alg(u_im, u_i, u_ip, x, h, b, c):
    return -((u_ip-2*u_i+u_im)/(h**2))+b*((u_ip-u_im)/(2*h))+c

def solution_u_alg(x):
    return -np.exp(x)


alpha = solution_u_alg(0)
beta = solution_u_alg(1)


# Für die Extrapolation später
solutions_alg = []

plt.figure(figsize=FIGSIZE)


print()
print("Allgemeiner Fall")
print("----------------------------------------")
print()

for N in N_values:
    x_numeric = np.linspace(0,1,N+1)
    # Der Startwert ist der lineare Interpolant der Werte alpha und Beta auf dem Gitter x
    u_initial = alpha + (beta - alpha)*x_numeric
    u_solution_numerical = newton(b_alg, dc, c_alg, u_initial, f_alg, N+1, alpha, beta)
    # In die Liste aufnehmen 
    solutions_alg.append(u_solution_numerical)
    # Numerische Lösung für momentanes N plotten 
    plt.plot(x_numeric, u_solution_numerical, label=f'Numerische Lösung (N={N})', linestyle='--', marker='o')

# Analytische Lösung plotten 
plt.plot(x_numeric, solution_u_alg(x_numeric), label='Analytische Lösung', linestyle='-', color='red', linewidth=2)

# Plot formatieren
plt.title('Vergleich Numerische Lösung für verschiedene N mit Analytischer Lösung (Allgemein)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()




# ERROR ANALYSE
# ---------------------------------------------------------------------------

max_errors = []
plt.figure(figsize=FIGSIZE)

# Berechnet den Fehler und plottet Ergebnisse 
for N in N_values:
    x_numeric = np.linspace(0, 1, N+1)
    u_initial = alpha + (beta - alpha)*x_numeric
    u_solution_numerical = newton(b_alg, dc, c_alg, u_initial, f_alg, N+1, alpha, beta)
    # Konstruierte Lösung an Stützpunkten berechnen 
    analytical_solution = solution_u_alg(x_numeric) 

    # Absoluten Fehler berechnen 
    error = np.abs(analytical_solution - u_solution_numerical)
    # Davon den maximalen Wert bestimmen und in die Liste aufnehmen 
    max_error_momentan = np.max(error)
    max_errors.append(max_error_momentan)



plt.plot(N_values, max_errors, marker='o', linestyle='-', color='blue', markersize=8, label='Global Max Error')


# Plot formatieren
# Doppelt logarithmische Darstellung
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error vs. N (Allgemeiner Fall)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()  
plt.show()


# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = {
        "N": N_values,
        "Max Error": max_errors,
        }

df = pd.DataFrame(data)

csv_file = "error_table_ohne_extrapolation_algemein.csv"
df.to_csv(csv_file, index=False)


print()
print("----------------------------------------")
print()
# Tabelle ausgeben 
print()
print()
print(df.to_string(index=False))
print()
print()




# EXTRAPOLATION
# -------------------------------------------------------------------------

# Erster Durchlauf
Extrapolierte_solution_alg_1 = Richardson_Extrapolation(solutions_alg, N_values, alpha, beta, 2)
# Zweiter Durchlauf
Extrapolierte_solution_alg_2 = Richardson_Extrapolation(Extrapolierte_solution_alg_1, N_values[0:-1], alpha, beta, 4)


i = 0

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:
    x_numeric = np.linspace(0, 1, N + 1)
    plt.plot(x_numeric, Extrapolierte_solution_alg_1[i], label=f'Extrapolierte Lösung (N={N})', marker='o', linestyle='--')
    i += 1


plt.plot(x_numeric, solution_u_alg(x_numeric), label='Analytische Lösung', color='black', linewidth=2)

plt.title('Vergleich Extrapolierte und Analytische Lösung (Allgemeiner Fall)')
plt.xlabel('x')
plt.ylabel('Extrapoliertes u(x)')
plt.legend()
plt.grid()
plt.show()


i = 0
# Enthält später die maximalen Fehler nach einem Extrapolationsdurchlauf 
max_errors_extrapolation = []

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-1]:

    x_numeric = np.linspace(0,1,N+1)
    # Konstruierte Lösung an Stützpunkten berechnen
    analytical_solution = solution_u_alg(x_numeric) 
    # Absoluten Fehler berrechnen 
    error = np.abs(analytical_solution - Extrapolierte_solution_alg_1[i])
    # Davon den maximalen Wert bestimmen und in die Liste aufnehmen 
    max_error_momentan = np.max(error)
    max_errors_extrapolation.append(max_error_momentan)
    i += 1


i = 0
# Enthält später die maximalen Fehler nach dem zweiten Extrapolationsdurchlauf 
max_errors_extrapolation_2 = []

plt.figure(figsize=FIGSIZE)

for N in N_values[0:-2]:

    x_numeric = np.linspace(0,1,N+1)
    # Konstruierte Lösung an Stützpunkten berechnen
    analytical_solution = solution_u_alg(x_numeric)
    # Absoluten Fehler berechnen
    error = np.abs(analytical_solution - Extrapolierte_solution_alg_2[i])
    # Davon den maximalen Wert bestimmen und in die Liste aufnehmen 
    max_error_momentan = np.max(error)
    max_errors_extrapolation_2.append(max_error_momentan)
    i += 1


# Maximalen Fehler für Verfahren ohne Extrapolation, nach einem Durchlauf und nach dem zweiten plotten 
plt.plot(N_values[0:-2], max_errors[0:-2], marker='o', linestyle='--', color='blue', markersize=8, label='Ohne Extrapolation')
plt.plot(N_values[0:-2], max_errors_extrapolation_1[0:-1], marker='o', linestyle='-', color='orange', markersize=8, label='Mit Extrapolation (Erster Durchlauf)')
plt.plot(N_values[0:-2], max_errors_extrapolation_2, marker='o', linestyle='-', color='red', markersize=8, label='Mit Extrapolation (Zweiter Durchlauf)')


# Plot formatieren
# Doppelt logarithmische Darstellung
plt.xscale('log')
plt.yscale('log')
plt.title('Globaler maximaler Error ohne und mit Extrapolation vs. N (Allgemeiner Fall)')
plt.xlabel('N')
plt.ylabel('Globaler maximaler Error')
plt.grid(True)
plt.legend()
plt.show()


# Erstelle Tabelle mit Fehlern und speicher diese extern ab
data = { 
        "N": N_values[0:-2],
        "Max Error (Ohne Extrapolation)": max_errors[0:-2],
        "Max Error (Mit Extrapolation (1. Durchlauf))": max_errors_extrapolation[0:-1],
        "Max Error (Mit Extrapolation (2. Durchlauf))": max_errors_extrapolation_2
    }

df = pd.DataFrame(data)

csv_file = "error_table_mit_extrapolation_algemein.csv"
df.to_csv(csv_file, index=False)


print("Allgemeiner Fall nach Extrapolation")
print("----------------------------------------")
print()
# Tabelle ausgeben 
print()
print()
print(df.to_string(index=False))
print()
print()



# --------------------------------------------------------------------------




