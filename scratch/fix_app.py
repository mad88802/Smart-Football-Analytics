"""
Script to fix the corrupted analyze_afd section in app.py.
The file currently has the 'chat' route body embedded inside analyze_afd,
after the AFD PLOT ERROR handling block.
We need to:
1. Replace everything after the plot error block (up to the launch_tableau route)
   with the proper matrix_html + afd_steps + render_template code
2. Re-insert the get_interpretation and chat routes
"""

with open(r'c:\Users\DZ Laptops\Desktop\ACP\app.py', 'rb') as f:
    content = f.read().decode('utf-8')

# Identify the boundaries
# After the plot error return statement, we need to insert the rest of analyze_afd
# then the missing routes

# The corrupted section starts right after the plot error handling
ANCHOR_BEFORE = '        return f"Erreur lors de la cr\u00e9ation du graphique AFD: {e}<br><pre>{tb}</pre>"\n\n'
ANCHOR_AFTER = '@app.route("/launch_tableau", methods=["POST"])'

i_start = content.find(ANCHOR_BEFORE)
i_end = content.find(ANCHOR_AFTER)

if i_start == -1:
    print("ERROR: Could not find ANCHOR_BEFORE")
    exit(1)
if i_end == -1:
    print("ERROR: Could not find ANCHOR_AFTER")
    exit(1)

print(f"Found section to replace: chars {i_start + len(ANCHOR_BEFORE)} to {i_end}")
print("Section being replaced:")
print(repr(content[i_start + len(ANCHOR_BEFORE):i_end]))
print("---")

# The replacement content (goes between the plot error and launch_tableau route)
REPLACEMENT = '''    def matrix_html(values, index=None, columns=None, decimals=4):
        try:
            return pd.DataFrame(values, index=index, columns=columns).round(decimals).to_html(classes='table', border=0)
        except Exception as e:
            return f"<p class=\\'text-secondary\\'>[Erreur affichage matrice: {e}]</p>"

    try:
        ld_cols = [c for c in afd_train["df_afd"].columns if c.startswith('LD')]
        afd_steps = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(afd_train["feature_names"]),
            "n_classes": len(afd_train["classes"]),
            "q_axes": len(ld_cols),
            "class_counts": afd_train["class_counts"],
            "X_html": afd_train["X_df"].head(100).round(4).to_html(classes='table', border=0),
            "y_html": afd_train["y_series"].head(100).to_frame().to_html(classes='table', border=0),
            "global_mean_html": afd_train["global_mean"].to_frame("Centre global").T.round(4).to_html(classes='table', border=0),
            "class_means_html": afd_train["class_means"].round(4).to_html(classes='table', border=0),
            "S_W_html": matrix_html(afd_train["S_W"], afd_train["feature_names"], afd_train["feature_names"]),
            "S_B_html": matrix_html(afd_train["S_B"], afd_train["feature_names"], afd_train["feature_names"]),
            "S_W_reg_html": matrix_html(afd_train["S_W_reg"], afd_train["feature_names"], afd_train["feature_names"]),
            "W_inv_html": matrix_html(afd_train["W_inv"], afd_train["feature_names"], afd_train["feature_names"]),
            "fisher_matrix_html": matrix_html(afd_train["fisher_matrix"], afd_train["feature_names"], afd_train["feature_names"]),
            "eig_vals_all_html": pd.DataFrame({
                "Axe": [f"LD{i+1}" for i in range(len(afd_train["eig_vals_all"]))],
                "Valeur propre": afd_train["eig_vals_all"]
            }).round(6).to_html(classes='table', border=0, index=False),
            "axes_html": matrix_html(afd_train["W_axes"], afd_train["feature_names"], ld_cols, decimals=6),
            "projection_html": afd_train["df_afd"].head(100).round(4).to_html(classes='table', border=0)
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("AFD STEPS BUILD ERROR:", tb)
        return f"Erreur lors de la construction des etapes AFD: {e}"

    try:
        return render_template("results_afd.html",
                               metrics=metrics,
                               afd_plot=fig_afd.to_json(),
                               classes=afd_train["classes"],
                               target_col=target_col,
                               initial_df=df.head(100).to_html(classes='table', border=0),
                               S_W=afd_train["S_W"],
                               S_B=afd_train["S_B"],
                               eig_vals=afd_train["eig_vals"],
                               eig_vecs=afd_train["eig_vecs"],
                               feature_names=afd_train["feature_names"],
                               afd_steps=afd_steps)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("AFD RENDER ERROR:", tb)
        return f"Erreur lors du rendu du template AFD: {e}"


@app.route("/get_interpretation", methods=["POST"])
def get_interpretation():
    data = request.json
    try:
        eig_percent = np.array(data['eig_percent'])
        coord_var = pd.DataFrame(data['coord_var'])
        C_vals = pd.DataFrame(data['C'])
        clusters = data.get('clusters', [])
        api_key = os.getenv('GROQ_API_KEY')
        interpretation = generate_llm_interpretation(eig_percent, coord_var, C_vals, clusters, api_key)
        return jsonify({"interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    chat_history = data.get('history', [])
    interpretation_data = data.get('interpretationData', {})
    user_message = data.get('message', '')
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        return jsonify({"error": "Cle API Groq manquante."}), 400

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        context_str = json.dumps({
            "eig_percent": interpretation_data.get("eig_percent"),
            "coord_var": interpretation_data.get("coord_var")
        }, ensure_ascii=False)

        system_prompt = f"""
Tu es un data scientist expert en Analyse en Composantes Principales (ACP) et un assistant de ce projet.
Reponds aux questions de l\'utilisateur de maniere concise, professionnelle, et en francais.
Voici le contexte partiel des resultats actuels (valeurs propres en pourcentages, et coordonnees des variables sur les axes):
{context_str}
Ne donne ces informations brutes que si elles sont utiles pour repondre a la question de l\'utilisateur.
Formate toujours ta reponse en markdown (utilisation de **gras**, de listes, etc. si necessaire).
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        model_names = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        last_error = ""
        for model_name in model_names:
            try:
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                return jsonify({"response": chat_completion.choices[0].message.content})
            except Exception as e:
                last_error = str(e)
                if "400" in last_error or "model_decommissioned" in last_error or "404" in last_error:
                    continue
                else:
                    break
        return jsonify({"error": f"Erreur avec les modeles Groq: {last_error}"}), 500

    except Exception as e:
        print(f"DEBUG ERROR GROQ CHAT: {str(e)}")
        return jsonify({"error": str(e)}), 500


'''

# Build the new content
new_content = (
    content[:i_start + len(ANCHOR_BEFORE)]
    + REPLACEMENT
    + content[i_end:]
)

with open(r'c:\Users\DZ Laptops\Desktop\ACP\app.py', 'wb') as f:
    f.write(new_content.encode('utf-8'))

print("Successfully patched app.py")
print(f"New file size: {len(new_content)} chars")
