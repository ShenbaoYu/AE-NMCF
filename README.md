# AE-NMCF
Student cognitive modeling (SCM) is a fundamental task in intelligent education, with applications ranging from personalized learning to educational resource allocation. By exploiting students' response logs, SCM aims to predict their exercise performance as well as estimate knowledge proficiency in a subject. Data mining approaches such as matrix factorization can obtain high accuracy in predicting student performance on exercises, but the knowledge proficiency is unknown or poorly estimated. The situation is further exacerbated if only sparse interactions exist between exercises and students (or knowledge concepts). To solve this dilemma, we root monotonicity -- a fundamental psychometric theory on educational assessments -- in a co-factorization framework and present an autoencoder-like nonnegative matrix co-factorization (AE-NMCF), which jointly factorizes student-exercise interactions and exercise-knowledge linkages to improve the accuracy of estimating the student's knowledge proficiency. The resulting estimation problem is nonconvex with nonnegative constraints. We introduce a projected gradient method based on block coordinate descent with Lipschitz constants and guarantee the method's theoretical convergence. Experiments on several real-world data sets demonstrate the efficacy of our approach in terms of both performance prediction accuracy and knowledge estimation ability, when compared with existing student cognitive models.
