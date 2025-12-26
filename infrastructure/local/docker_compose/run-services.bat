@echo off
REM Helper script to run services with docker-compose profiles
REM
REM Usage:
REM   run-services.bat infra              # Only infrastructure (Kafka, DBs)
REM   run-services.bat preprocess         # Infrastructure + preprocess service
REM   run-services.bat ingest             # Infrastructure + ingest service
REM   run-services.bat all                # Everything
REM   run-services.bat down               # Stop all services
REM   run-services.bat restart <profile>  # Restart specific profile
REM   run-services.bat logs <service>     # Tail logs for specific service
REM   run-services.bat ps                 # Show running services

setlocal

set COMPOSE_FILE=docker-compose.yml
set COMPOSE_PROJECT=ai-trading
cd /d "%~dp0"

if "%1"=="" set "PROFILE=infra"
if not "%1"=="" set "PROFILE=%1"

if "%PROFILE%"=="infra" (
    echo Starting infrastructure only (Kafka, TimescaleDB, Postgres^)...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% down 2>nul
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% up -d
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% logs -f
    goto :end
)

if "%PROFILE%"=="preprocess" (
    echo Starting infrastructure + preprocess service...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile preprocess down 2>nul
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile preprocess up -d --build
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile preprocess logs -f
    goto :end
)

if "%PROFILE%"=="ingest" (
    echo Starting infrastructure + ingest service...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile ingest down 2>nul
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile ingest up -d --build
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile ingest logs -f
    goto :end
)

if "%PROFILE%"=="all" (
    echo Starting all services...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile all down 2>nul
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile all up -d --build
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile all logs -f
    goto :end
)

if "%PROFILE%"=="restart" (
    if "%2"=="" (
        echo Usage: %0 restart ^<profile^>
        exit /b 1
    )
    echo Restarting %2...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile %2 down
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile %2 up -d --build
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile %2 logs -f
    goto :end
)

if "%PROFILE%"=="logs" (
    if "%2"=="" (
        echo Showing all logs...
        docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% logs -f
    ) else (
        echo Showing logs for %2...
        docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% logs -f %2
    )
    goto :end
)

if "%PROFILE%"=="ps" (
    echo Showing running services...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% ps
    goto :end
)

if "%PROFILE%"=="down" (
    echo Stopping all services...
    docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile all down
    goto :end
)

if "%PROFILE%"=="clean" (
    echo Stopping all services and removing volumes...
    set /p CONFIRM="This will delete all data. Are you sure? (y/N): "
    if /i "%CONFIRM%"=="y" (
        docker-compose -p %COMPOSE_PROJECT% -f %COMPOSE_FILE% --profile all down -v
    )
    goto :end
)

echo Usage: %0 {infra^|preprocess^|ingest^|all^|down^|restart^|logs^|ps^|clean}
echo.
echo Commands:
echo   infra              - Start infrastructure only
echo   preprocess         - Start infrastructure + preprocess service
echo   ingest             - Start infrastructure + ingest service
echo   all                - Start all services
echo   restart ^<profile^>  - Restart specific profile
echo   logs [service]     - Show logs (optional: for specific service^)
echo   ps                 - Show running services
echo   down               - Stop all services
echo   clean              - Stop all and remove volumes (data loss!^)
exit /b 1

:end
endlocal
