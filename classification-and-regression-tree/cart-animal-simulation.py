from sklearn.tree import DecisionTreeClassifier, export_text

# 1. THE CLUES (Features)
# Inside the brackets: [Has Wings?, Breathes Fire?]
# 0 means NO, 1 means YES
clues = [[0, 0], [1, 0], [1, 1], [0, 1]]  # Animal 1: No wings, No fire (This is a Dog),  # Animal 2: Has wings, No fire (This is a Parrot),  # Animal 3: Has wings, Breathes fire (This is a Dragon),  # Animal 4: No wings, Breathes fire (This is a Mystery Animal)


# 2. THE SECRET ANSWERS (Labels)
# Let's give each animal a nickname code
# "Dog" = 0, "Parrot" = 1, "Dragon" = 2, "Unknown" = 3
secret_answers = [0, 1, 2, 3]

# 3. CREATE THE SMART ROBOT (The Tree Model)
# We open a blank decision tree game
robot_brain = DecisionTreeClassifier()

# 4. TIME TO STUDY! (Training)
# The robot looks at the clues and learns the answers
robot_brain.fit(clues, secret_answers)

print("The Robot Brain has learned the animal rules! \n")

# 5. TEST THE ROBOT WITH A MYSTERY ANIMAL!
# Let's give it a mystery creature that HAS WINGS (1) and BREATHES FIRE (1)
mystery_animal = [[1, 1]]
prediction = robot_brain.predict(mystery_animal)

# 6. REVEAL THE GUESS!
if prediction == 0:
    print("Robot says: 'Woof! That must be a DOG!' ")
elif prediction == 1:
    print("Robot says: 'Squawk! That must be a PARROT!' ")
elif prediction == 2:
    print("Robot says: 'ROAR! That is a magical DRAGON!' ")
elif prediction == 3:
    print("Robot says: ❌ Error! That animal makes no sense!")

# 7. SEE THE ROBOT'S INNER FLOWCHART
print("\n---  Look inside the Robot's Secret Map ---")
print(
    export_text(
        robot_brain, feature_names=["Has Wings?", "Breathes Fire?"]
    )
)
