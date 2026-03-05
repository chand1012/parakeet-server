use rocket::data::Data;
use rocket::{get, launch, post, routes, serde::json::Json, serde::Serialize, Build, Rocket};

#[derive(Serialize)]
struct Message<'r> {
    greeting: String,
    name: &'r str,
    age: u8,
}

#[get("/hello/<name>/<age>")]
fn hello(name: &str, age: u8) -> Json<Message<'_>> {
    Json(Message {
        greeting: "Hello".to_string(),
        name,
        age,
    })
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[post(
    "/v1/audio/transcriptions",
    format = "multipart/form-data",
    data = "<_guard>"
)]
fn transcribe(_guard: Data<'_>) -> Json<TranscriptionResponse> {
    Json(TranscriptionResponse {
        text: "This is the transcribed text from the audio file.".to_string(),
    })
}

#[launch]
fn rocket() -> Rocket<Build> {
    rocket::build().mount("/", routes![hello, transcribe])
}
