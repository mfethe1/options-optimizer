"""
Nightly Eval Job for OpenAI Evals Test Suite

Automated execution of evals with:
- Eval execution
- Results storage
- Alerts for failures (<95% pass rate)
- Trend tracking over time
- Email/Slack notifications (optional)

Usage:
    python scripts/nightly_eval_job.py
    python scripts/nightly_eval_job.py --alert-email user@example.com
    python scripts/nightly_eval_job.py --alert-slack https://hooks.slack.com/...
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_evals import EvalRunner


class NightlyEvalJob:
    """Nightly eval job runner"""
    
    def __init__(self, alert_email: str = None, alert_slack: str = None):
        self.alert_email = alert_email
        self.alert_slack = alert_slack
        self.runner = EvalRunner()
    
    async def run_evals(self) -> dict:
        """Run evals and return summary"""
        print(f"🌙 Starting nightly eval job at {datetime.now().isoformat()}\n")
        
        summary = await self.runner.run_all_evals()
        
        return summary
    
    def check_pass_rate(self, summary: dict) -> bool:
        """Check if pass rate meets target"""
        return summary['met_target']
    
    def send_email_alert(self, summary: dict):
        """Send email alert for failures"""
        if not self.alert_email:
            return
        
        print(f"\n📧 Sending email alert to {self.alert_email}...")
        
        # In production, use smtplib or SendGrid
        subject = f"⚠️ Eval Failure Alert: {summary['pass_rate']:.1%} pass rate"
        body = f"""
Nightly Eval Job Failed
=======================

Timestamp: {summary['timestamp']}
Total cases: {summary['total_cases']}
Passed: {summary['passed']}
Failed: {summary['failed']}
Pass rate: {summary['pass_rate']:.1%} (target: {summary['target_pass_rate']:.1%})
Average score: {summary['average_score']:.2f}/5.0

Failed cases:
"""
        
        for result in summary['results']:
            if not result['passed']:
                body += f"\n- {result['case_id']}: {result['overall_score']:.2f}/5.0"
        
        body += "\n\nPlease investigate and fix the failing cases."
        
        print(f"Subject: {subject}")
        print(f"Body:\n{body}")
        print("✅ Email alert sent (mock)")
    
    def send_slack_alert(self, summary: dict):
        """Send Slack alert for failures"""
        if not self.alert_slack:
            return
        
        print(f"\n💬 Sending Slack alert to {self.alert_slack}...")
        
        # In production, use requests to post to Slack webhook
        message = {
            "text": f"⚠️ Eval Failure Alert: {summary['pass_rate']:.1%} pass rate",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "⚠️ Nightly Eval Job Failed"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Timestamp:*\n{summary['timestamp']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Pass Rate:*\n{summary['pass_rate']:.1%} (target: {summary['target_pass_rate']:.1%})"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Cases:*\n{summary['total_cases']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Failed:*\n{summary['failed']}"
                        }
                    ]
                }
            ]
        }
        
        print(f"Message: {json.dumps(message, indent=2)}")
        print("✅ Slack alert sent (mock)")
    
    def track_trends(self, summary: dict):
        """Track eval trends over time"""
        print("\n📊 Tracking eval trends...")
        
        trends_file = Path("eval_results") / "trends.json"
        
        # Load existing trends
        if trends_file.exists():
            with open(trends_file) as f:
                trends = json.load(f)
        else:
            trends = {
                'history': []
            }
        
        # Add current summary
        trends['history'].append({
            'timestamp': summary['timestamp'],
            'pass_rate': summary['pass_rate'],
            'average_score': summary['average_score'],
            'total_cases': summary['total_cases'],
            'passed': summary['passed'],
            'failed': summary['failed']
        })
        
        # Keep only last 30 days
        if len(trends['history']) > 30:
            trends['history'] = trends['history'][-30:]
        
        # Calculate trend statistics
        if len(trends['history']) >= 2:
            recent_pass_rates = [h['pass_rate'] for h in trends['history'][-7:]]
            trends['7_day_avg_pass_rate'] = sum(recent_pass_rates) / len(recent_pass_rates)
            
            recent_scores = [h['average_score'] for h in trends['history'][-7:]]
            trends['7_day_avg_score'] = sum(recent_scores) / len(recent_scores)
        
        # Save trends
        with open(trends_file, 'w') as f:
            json.dump(trends, f, indent=2)
        
        print(f"✅ Trends saved to {trends_file}")
        
        if len(trends['history']) >= 2:
            print(f"  7-day avg pass rate: {trends['7_day_avg_pass_rate']:.1%}")
            print(f"  7-day avg score: {trends['7_day_avg_score']:.2f}/5.0")
    
    async def run(self):
        """Run nightly eval job"""
        try:
            # Run evals
            summary = await self.run_evals()
            
            # Track trends
            self.track_trends(summary)
            
            # Check pass rate and send alerts if needed
            if not self.check_pass_rate(summary):
                print(f"\n⚠️ ALERT: Pass rate {summary['pass_rate']:.1%} below target {summary['target_pass_rate']:.1%}")
                self.send_email_alert(summary)
                self.send_slack_alert(summary)
                return 1  # Exit code 1 for failure
            else:
                print(f"\n✅ SUCCESS: Pass rate {summary['pass_rate']:.1%} meets target {summary['target_pass_rate']:.1%}")
                return 0  # Exit code 0 for success
        
        except Exception as e:
            print(f"\n❌ ERROR: Nightly eval job failed with exception: {e}")
            
            # Send critical alert
            if self.alert_email:
                print(f"📧 Sending critical alert to {self.alert_email}...")
            if self.alert_slack:
                print(f"💬 Sending critical alert to {self.alert_slack}...")
            
            return 2  # Exit code 2 for critical failure


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Nightly eval job for OpenAI Evals Test Suite")
    parser.add_argument("--alert-email", help="Email address for failure alerts")
    parser.add_argument("--alert-slack", help="Slack webhook URL for failure alerts")
    
    args = parser.parse_args()
    
    job = NightlyEvalJob(
        alert_email=args.alert_email,
        alert_slack=args.alert_slack
    )
    
    exit_code = await job.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

