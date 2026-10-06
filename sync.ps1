<#
.SYNOPSIS
    Automated Git Sync Script for Problem 1 GPU Repository
.DESCRIPTION
    Pulls latest remote changes, stages any local modifications, and pushes to GitHub.
#>

Write-Host "🔄 Fetching and syncing with GitHub (origin/main)..." -ForegroundColor Cyan

# 1. Fetch remote updates
git fetch origin main

# 2. Rebase or pull remote commits
git pull --rebase origin main

# 3. Stage any modified or new files
git add -A

# 4. Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    $commitMsg = "Sync update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "📦 Committing local changes: $commitMsg" -ForegroundColor Yellow
    git commit -m "$commitMsg"
} else {
    Write-Host "✅ No uncommitted local changes." -ForegroundColor Green
}

# 5. Push commits to GitHub
Write-Host "🚀 Pushing commits to GitHub..." -ForegroundColor Cyan
git push origin main

Write-Host "🎉 Repository is fully in sync!" -ForegroundColor Green
