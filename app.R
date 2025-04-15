library(shiny)

# Load the trained model
model <- readRDS("logistic_model.rds")
threshold <- 0.35

ui <- fluidPage(
  titlePanel("🩺 Heart Disease Risk Predictor"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("age", "Age Grp", min = 0, max = 15, value = 5),
      sliderInput("bmi", "BMI", min = 10, max = 50, value = 25),
      sliderInput("mental_health", "Mental Health Days", min = 0, max = 30, value = 0),
      sliderInput("physical_health", "Physical Health Days", min = 0, max = 30, value = 0),
      sliderInput("gen_hlth", "General Health (1=Excellent, 5=Poor)", 1, 5, 3),
      selectInput("education", "Education Level", 1:6),
      selectInput("income", "Income Level", 1:8),
      checkboxInput("high_bp", "High Blood Pressure?", FALSE),
      checkboxInput("high_chol", "High Cholesterol?", FALSE),
      checkboxInput("chol_check", "Cholesterol Check in last 5 years?", FALSE),
      checkboxInput("smoker", "Smoker?", FALSE),
      checkboxInput("stroke", "Ever had a Stroke?", FALSE),
      checkboxInput("diabetes", "Have Diabetes?", FALSE),
      checkboxInput("phys_activity", "Physically Active?", FALSE),
      checkboxInput("diff_walk", "Difficulty Walking?", FALSE),
      checkboxInput("sex", "Male?", FALSE),
      checkboxInput("no_doc_cost", "Could not see doctor due to cost?", FALSE),
      checkboxInput("any_healthcare", "Have healthcare coverage?", FALSE),
      checkboxInput("hvy_alcohol", "Heavy Alcohol Consumption?", FALSE),
      actionButton("predict", "Predict")
    ),

    mainPanel(
      verbatimTextOutput("result"),
      verbatimTextOutput("probability")
    )
  )
)

server <- function(input, output) {
  observeEvent(input$predict, {
    # Create input vector in the same order as training
    new_data <- data.frame(
      HighBP = as.integer(input$high_bp),
      HighChol = as.integer(input$high_chol),
      CholCheck = as.integer(input$chol_check),
      BMI = input$bmi,
      Smoker = as.integer(input$smoker),
      Stroke = as.integer(input$stroke),
      Diabetes = as.integer(input$diabetes),
      PhysActivity = as.integer(input$phys_activity),
      Fruits = 0,      # Placeholder (no UI yet)
      Veggies = 0,     # Placeholder (no UI yet)
      HvyAlcoholConsump = as.integer(input$hvy_alcohol),
      AnyHealthcare = as.integer(input$any_healthcare),
      NoDocbcCost = as.integer(input$no_doc_cost),
      GenHlth = input$gen_hlth,
      MentHlth = input$mental_health,
      PhysHlth = input$physical_health,
      DiffWalk = as.integer(input$diff_walk),
      Sex = as.integer(input$sex),
      Age = input$age,
      Education = as.integer(input$education),
      Income = as.integer(input$income)
    )

    # Predict probability
    prob <- predict(model, newdata = new_data, type = "response")
    prediction <- ifelse(prob >= threshold, "⚠️ High risk detected. Recommend further medical evaluation.", "✅ Low risk detected. Keep up the healthy habits!")

    output$result <- renderText({
      paste("Prediction:", prediction)
    })

    output$probability <- renderText({
      paste("Predicted probability:", round(prob * 100, 2), "%")
    })
  })
}

shinyApp(ui = ui, server = server)
