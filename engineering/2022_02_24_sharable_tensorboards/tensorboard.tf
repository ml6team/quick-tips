# Google Cloud Storage

resource "google_storage_bucket" "tensorboard_logs_bucket" {
  name     = "${var.project}-tensorboard-logs"
  project  = var.project
  location = var.region
}

resource "google_storage_bucket_object" "requirements" {
  name    = "requirements.txt"
  content = "tensorflow==2.7"
  bucket  = google_storage_bucket.tensorboard_logs_bucket.name
}

# IAP

resource "google_project_service" "iap" {
  project = var.project
  service = "iap.googleapis.com"
}

resource "google_iap_brand" "oauth_consent_screen" {
  support_email     = "example@mail.com"
  application_title = "Tensorboard"
  depends_on        = [google_project_service.iap]
}

# App Engine

resource "google_app_engine_application" "app" {
  project     = var.project
  location_id = "europe-west"
  iap {
    enabled              = true
    oauth2_client_id     = google_iap_client.oauth_client.client_id
    oauth2_client_secret = google_iap_client.oauth_client.secret
  }

  depends_on = [google_project_service.iap]
}

resource "google_app_engine_standard_app_version" "tensorboard" {
  service    = "tensorboard"
  runtime    = "python39"
  version_id = formatdate("YYYYMMDDhhmmss", timestamp())

  entrypoint {
    shell = "tensorboard --logdir $LOG_DIR --host $HOST --port $PORT --load_fast $LOAD_FAST"
  }

  deployment {
    files {
      name       = "requirements.txt"
      source_url = "https://storage.googleapis.com/${google_storage_bucket.tensorboard_logs_bucket.name}/${google_storage_bucket_object.requirements.name}"
    }
  }

  env_variables = {
    LOG_DIR   = "gs://${google_storage_bucket.tensorboard_logs_bucket.name}"
    PORT      = "8080"
    HOST      = "0.0.0.0"
    LOAD_FAST = "false"
  }

  instance_class = "F4"

  handlers {
    url_regex = ".*"
    script {
      script_path = "auto"
    }
    login            = "LOGIN_REQUIRED"
    auth_fail_action = "AUTH_FAIL_ACTION_UNAUTHORIZED"
  }

  timeouts {
    create = "10m"
  }

  depends_on = [google_project_service.iap, google_storage_bucket_object.requirements]

  delete_service_on_destroy = true
}

resource "google_iap_client" "oauth_client" {
  display_name = "IAP-App-Engine-app"
  brand        = google_iap_brand.oauth_consent_screen.name
  depends_on   = [google_project_service.iap]
}

# Allowed users

resource "google_iap_web_iam_binding" "allowed_users" {
  project = var.project
  role    = "roles/iap.httpsResourceAccessor"
  members = [
    "domain:your-domain.com",
  ]
}