# -Interaction-Stepwise-FDR-analysis

Python haven't built a stepwise regression package till 2019. 
The code here provides a function to run stepwise on logistic regression and uses AIC to evaluate the model.
  
--------------------------------------------------------------  
      from sklearn.linear_model import LogisticRegression
      import statsmodels.api as sm

      def forward_selected(data, response):
        """ This function is modified based on 
          https://planspace.org/20150423-forward_selection_with_statsmodels/

          Linear model designed by forward selection.
        """

          # Get column names and assign them to "remaining" (unselected variables)
        remaining = set(data.columns)

          # Remove Y from remaining
        remaining.remove(response)

          # Create an empty list to store selected variables
        selected = []

          # Set current AIC score to 1200 and best AIC score to 0
        current_score, best_new_score = 1200.0, 0.0

          # While remaining is not empty and current score >= best score
        while remaining and current_score >= best_new_score:

              # Create an empty list for accuracy scores
            scores_with_candidates = []

              # For each candidate variable in the unselected variables, 
              # fit logistic regression on selected variables + candidate
            for candidate in remaining:
                y_train = data[response]
                x_train = data[selected + [candidate]]
                log_reg = sm.Logit(y_train, x_train).fit()
                score = log_reg.aic

                  # Update the score list with new score and candidate variable
                scores_with_candidates.append((score, candidate))

              # Sort the score list in ascending order
            scores_with_candidates.sort(reverse=True)
              # Get the highest score and the corresponding variable
            best_new_score, best_candidate = scores_with_candidates.pop()
              # If current score >= best new score, select this candidate variable
              # remove this variable from remaining
              # Update the current score
            if current_score >= best_new_score:

                  # Remove the candidate from remaining
                remaining.remove(best_candidate)

                  # Add the candidate to selected
                selected.append(best_candidate)

                  # Replace the current score with the best score
                current_score = best_new_score
        return selected
